import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np # Import numpy for np.nan

# URLs for IPO data
SUBSCRIPTION_URL = "https://ipowatch.in/ipo-subscription-status-today/"
GMP_URL = "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"

def fetch_and_parse_table(url):
    """
    Fetches HTML from a given URL and parses the first table found.
    Returns headers and rows as a list of lists.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the first table on the page
    table = soup.find('table')
    if not table:
        print(f"No table found on {url}")
        return None, None

    # Modified header extraction: Look for strong tags within the first row's cells
    headers = []
    first_row = table.find('tr')
    if first_row:
        # Try to find headers in th tags first, then strong tags in td tags
        ths = first_row.find_all('th')
        if ths:
            headers = [th.get_text(strip=True) for th in ths]
        else:
            tds_in_first_row = first_row.find_all('td')
            # Extract text from strong tag if present, otherwise from td itself
            headers = [td.find('strong').get_text(strip=True) if td.find('strong') else td.get_text(strip=True) for td in tds_in_first_row]
    
    if not headers:
        print(f"Could not extract headers from {url}")
        return None, None
            
    rows_data = []
    # Start from the second row for data, as the first row is now explicitly handled for headers
    for row in table.find_all('tr')[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if cells: # Only add if row is not empty
            rows_data.append(cells)
            
    return headers, rows_data

def normalize_stock_name(name):
    """
    Normalizes stock names by removing 'Closed', 'Open', and extra spaces,
    and content in parentheses, using regex for robust matching.
    """
    if isinstance(name, str):
        # Remove ' Closed' or ' Open' with optional leading/trailing spaces, case-insensitive
        name = re.sub(r'\s*(Closed|Open)\s*', '', name, flags=re.IGNORECASE).strip()
        # Remove any content in parentheses (e.g., (SME))
        name = re.sub(r'\s*\(. *\)\s*', '', name).strip()
        name = name.strip() # Final strip in case of leading/trailing spaces after regex
    return name

def clean_numeric_column(series):
    """
    Cleans a pandas Series by removing non-numeric characters (like Rupee symbol, commas)
    and converting to a numeric type. Explicitly handles '-' and empty strings as NaN.
    """
    if not isinstance(series, pd.Series):
        return series # Return as is if not a Series (e.g., a single string)

    # Convert to string to ensure .str methods work
    cleaned_series = series.astype(str)

    # Replace specific non-numeric indicators of missing data with NaN
    cleaned_series = cleaned_series.replace('-', np.nan)
    cleaned_series = cleaned_series.replace(' ', np.nan) # Also replace empty spaces if any

    # Use regex to remove other common non-numeric characters (like Rupee symbol, commas, percentage signs)
    # The regex [^\d.-]+ means "one or more characters that are NOT a digit, a dot, or a minus sign"
    cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
    
    # Convert empty strings resulting from regex cleaning to NaN
    cleaned_series = cleaned_series.replace('', np.nan)

    # Finally, convert to float. Now `errors='coerce'` will turn any remaining non-convertible strings to NaN.
    return pd.to_numeric(cleaned_series, errors='coerce')

def main():
    # --- Fetch Subscription Data ---
    print(f"Fetching subscription data from {SUBSCRIPTION_URL}...")
    sub_headers, sub_rows = fetch_and_parse_table(SUBSCRIPTION_URL)

    if not sub_headers or not sub_rows:
        print("Failed to get subscription data. Exiting.")
        return

    # Ensure headers match the number of columns in rows
    max_cols_sub = max(len(row) for row in sub_rows)
    if len(sub_headers) < max_cols_sub:
        sub_headers.extend([f'Unnamed_{i}' for i in range(len(sub_headers), max_cols_sub)])
    elif len(sub_headers) > max_cols_sub:
        sub_headers = sub_headers[:max_cols_sub]

    sub_df = pd.DataFrame(sub_rows, columns=sub_headers)
    print("Subscription data fetched successfully.")
    
    if sub_headers and len(sub_headers) > 0:
        sub_ipo_col = sub_headers[0]
        if sub_ipo_col in sub_df.columns:
            sub_df['Normalized_IPO_Name'] = sub_df[sub_ipo_col].apply(normalize_stock_name)
        else:
            print(f"Warning: Column '{sub_ipo_col}' not found in subscription DataFrame. Cannot normalize IPO names.")
            return
    else:
        print("Warning: Subscription DataFrame has no headers. Cannot normalize IPO names.")
        return

    # Apply cleaning to relevant columns in subscription DataFrame
    for col in ['QIB', 'NII', 'RII', 'Total', 'Applications']:
        if col in sub_df.columns:
            sub_df[col] = clean_numeric_column(sub_df[col])

    # --- Fetch GMP Data ---
    print(f"\nFetching GMP data from {GMP_URL}...")
    gmp_headers, gmp_rows = fetch_and_parse_table(GMP_URL)
    if not gmp_headers or not gmp_rows:
        print("Failed to get GMP data. Exiting.")
        return

    # Ensure headers match the number of columns in rows for GMP
    max_cols_gmp = max(len(row) for row in gmp_rows)
    if len(gmp_headers) < max_cols_gmp:
        gmp_headers.extend([f'Unnamed_{i}' for i in range(len(gmp_headers), max_cols_gmp)])
    elif len(gmp_headers) > max_cols_gmp:
        gmp_headers = gmp_headers[:max_cols_gmp]

    gmp_headers = ['IPO GMP' if header == 'IPO GMP*' else header for header in gmp_headers]
    gmp_df = pd.DataFrame(gmp_rows, columns=gmp_headers)
    print("GMP data fetched successfully.")

    if gmp_headers and len(gmp_headers) > 0:
        gmp_ipo_col = gmp_headers[0]
        if gmp_ipo_col in gmp_df.columns:
            gmp_df['Normalized_IPO_Name'] = gmp_df[gmp_ipo_col].apply(normalize_stock_name)
        else:
            print(f"Warning: Column '{gmp_ipo_col}' not found in GMP DataFrame. Cannot normalize IPO names.")
            return
    else:
        print("Warning: GMP DataFrame has no headers. Cannot normalize IPO names.")
        return

    # Apply cleaning to relevant columns in GMP DataFrame
    for col in ['IPO GMP', 'IPO Price', 'Listing Gain']:
        if col in gmp_df.columns:
            gmp_df[col] = clean_numeric_column(gmp_df[col])
    
    print("Merging dataframes...")
    merged_df = pd.merge(sub_df, gmp_df, on='Normalized_IPO_Name', how='outer', suffixes=('_sub', '_gmp'))
    
    # Filter out entries with missing subscription data
    if 'Total' in merged_df.columns:
        merged_df.dropna(subset=['Total'], inplace=True)
        print("Filtered out entries with missing subscription data.")
    else:
        print("Warning: 'Total' column not found after merge. Skipping filtering for missing subscription data.")

    # Filter out records where 'IPO GMP' is null
    if 'IPO GMP' in merged_df.columns:
        merged_df.dropna(subset=['IPO GMP'], inplace=True)
        print("Filtered out entries with missing IPO GMP data.")
    else:
        print("Warning: 'IPO GMP' column not found after merge. Skipping filtering for missing GMP data.")

    print("\n--- Merged IPO Data ---")
    print(merged_df.to_string()) # Use to_string() to avoid truncation

    # Save to JSON file
    output_filename = "merged_ipo_data.json"
    merged_df.to_json(output_filename, orient="records", indent=4)
    print(f"\nMerged data saved to {output_filename}")

if __name__ == "__main__":
    main()
