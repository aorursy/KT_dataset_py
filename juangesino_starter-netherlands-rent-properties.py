# Import some necessary libraries
from IPython.display import display, Markdown, Latex
import json
import pandas as pd
import re
raw_data_file = "/kaggle/input/netherlands-rent-properties/properties.json"

def load_raw_data(filepath):
    raw_data = []
    for line in open(filepath, 'r'):
        raw_data.append(json.loads(line))
    
    return raw_data
    
raw_data = load_raw_data(raw_data_file)

Markdown(f"""
Successfully imported {len(raw_data)} properties from the dataset.
""")
df = pd.DataFrame(raw_data)

Markdown(f"""
Successfully created DataFrame with shape: {df.shape}.
""")
df.info()
# Define all columns that need to be flatten and the property to extract
flatten_mapper = {
    "_id": "$oid",
    "crawledAt": "$date",
    "firstSeenAt": "$date",
    "lastSeenAt": "$date",
    "detailsCrawledAt": "$date",
}

# Function to do all the work of flattening the columns using the mapper
def flatten_columns(df, mapper):
    
    # Iterate all columns from the mapper
    for column in flatten_mapper:
        prop = flatten_mapper[column]
        raw_column_name = f"{column}_raw"
        
        # Check if the raw column is already there
        if raw_column_name in df.columns:
            # Drop the generated one
            df.drop(columns=[column], inplace=True)
            
            # Rename the raw back to the original
            df.rename(columns={ raw_column_name: column }, inplace=True)        
    
        # To avoid conflicts if re-run, we will rename the columns we will change
        df.rename(columns={
            column: raw_column_name,
        }, inplace=True)

        # Get the value inside the dictionary
        df[column] = df[raw_column_name].apply(lambda obj: obj[prop])
        
    return df
        
df = df.pipe(flatten_columns, mapper=flatten_mapper)
def rename_columns(df):
    # Store a dictionary to be able to rename later
    rename_mapper = {}
    
    # snake_case REGEX pattern
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    
    # Iterate the DF's columns
    for column in df.columns:
        rename_mapper[column] = pattern.sub('_', column).lower()
        
    # Rename the columns using the mapper
    df.rename(columns=rename_mapper, inplace=True)
    
    return df
df = df.pipe(rename_columns)
def parse_types(df):
    
    df["crawled_at"] = pd.to_datetime(df["crawled_at"])
    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"])
    df["last_seen_at"] = pd.to_datetime(df["last_seen_at"])
    df["details_crawled_at"] = pd.to_datetime(df["details_crawled_at"])
    df["latitude"] = pd.to_numeric(df["latitude"])
    df["longitude"] = pd.to_numeric(df["longitude"])
    
    return df
df = df.pipe(parse_types)
raw_data = load_raw_data(raw_data_file)
df = pd.DataFrame(raw_data)
df = (df
      .pipe(flatten_columns, mapper=flatten_mapper)
      .pipe(rename_columns)
      .pipe(parse_types)
     )
# Flatten column with list of objects
def flatten_col_list(lst):
    return list(map(lambda obj: obj["$date"], lst))

# Transform the DF into a time series
def to_timeseries(df, dates_column="dates_published"):
    # Get a list of columns without the target column
    columns = df.columns.values.tolist()
    columns.remove(dates_column)
    
    # Create a DF with all the dates
    dates_df = pd.DataFrame(df[dates_column].apply(flatten_col_list).tolist())
    
    # Create a wide representation of our DF
    wide = pd.concat([df, dates_df], axis=1).drop(dates_column, axis=1)
    
    # Melt the dataframe
    ts = pd.melt(wide, id_vars=columns, value_name='date')
    
    # [WARNING] Drop columns with missing date
    ts.dropna(inplace=True, subset=["date"])
    
    # Parse the date column
    ts["date"] = pd.to_datetime(ts["date"])
    
    # Offset the date column to account for timezone differences
    ts["date"] = ts["date"] + pd.DateOffset(hours=3)
    
    return ts
ts = df[:100].pipe(to_timeseries)
# Get a random property to show the time series
target_external_id = ts["external_id"].sample().iloc[0]
ts[ts["external_id"] == target_external_id][["date", "external_id", "city"]].head(10)