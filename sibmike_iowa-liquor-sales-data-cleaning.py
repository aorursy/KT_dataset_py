# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc  # Garbage collection. We will use it a lot.



from tqdm.notebook import trange, tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load raw dataset and sample 10% of it right away:

raw_df = pd.read_csv('/kaggle/input/iowa-liquor-sales/Iowa_Liquor_Sales.csv', parse_dates=['Date']).sample(frac=0.1)

gc.collect()
# Show first 5 records:

raw_df.head(5)
# Info on columns:

raw_df.info()
object_column_list = list(raw_df.dtypes[raw_df.dtypes == object].index)

object_column_list.remove('Store Location')

# We will need Store Laocation later:

for object_column in object_column_list:

    raw_df.loc[:,object_column] = raw_df.loc[:,object_column].str.lower().str.strip().str.split().str.join(' ')

    print(object_column)

    gc.collect()
# Check if any columns can be converted to categories:

count_dict = {}

for c in tqdm(raw_df.columns):

    count_per_value = raw_df[c].value_counts()

    count_dict[c] = {

        'count_of_values': count_per_value.count(),

        'count_per_value': count_per_value

    }

    print(f"Column {c} has {count_dict[c]['count_of_values']} values.")
# Note: we are not using count_dict as we will use the function for batch processing new data:



def get_coordinates(location_series):

    

    # Get unique values:

    unique_locations = location_series.unique()

    

    # Create Pandas series in order to use .str methods:

    unique_loactions = pd.Series(unique_locations, index=unique_locations)

    

    # Split every series value using '\n'

    # Take the last element from the list 

    # Convert series to a dictionary

    # (Note: [-1:] trick this helps to dodge empty lists)

    location_dict = unique_loactions.str.split('\n').str[-1:].str[0].to_dict()



    # Use the dict to map Locations to Coordinates:

    return location_series.map(location_dict, na_action='ignore')



# Create a new column coordinates

raw_df.loc[:,'Coordinates'] = get_coordinates(raw_df['Store Location'])



# Get rid of useless column:

raw_df.drop(columns=['Store Location'], inplace=True)

gc.collect()



# Check 5 rows of new column:

raw_df.Coordinates[:5]
# Stores:

stores_df = raw_df.loc[:,['Store Number','Store Name','Address','Coordinates','City']].drop_duplicates()



# Count values for every store number:

store_number_counts = stores_df.loc[:,"Store Number"].value_counts()



# Find store counts with multiple name or address antries:

idx_store_duplicates = store_number_counts[store_number_counts!=1].index
# An example of multiple records:

stores_df.loc[stores_df["Store Number"]==idx_store_duplicates[5]].drop_duplicates()
# Process Store Name:

raw_df[['Store Name','Store Subname']] = raw_df['Store Name'].str.rsplit(pat=" / ", expand=True, n=1)

raw_df[['Store Name','Store SubNumber']] = raw_df['Store Name'].str.rsplit(pat=" #", expand=True, n=1)
# Lets do the sam trick for Vednors:

def max_length(df, number_column, name_column):

    # Stores:

    stores_df = df.loc[:,[number_column, name_column]].drop_duplicates()

    

    # Create dictionary to map CountyNumber to max CountyName

    max_dict = stores_df.fillna('#').groupby(number_column)[name_column].max().to_dict()

    

    # Replace CountyNames with max CountyNames using the dictionary:

    return raw_df[number_column].map(max_dict)
raw_df.loc[:,'Store Name'] = max_length(raw_df,'Store Number','Store Name')

raw_df.loc[:,'Store Subname'] = max_length(raw_df,'Store Number','Store Subname')
l = raw_df['Store Name'].str.extract(pat=r'(\d+$)')
raw_df.loc[:,'County'] = max_length(raw_df,'County Number', 'County')
raw_df.loc[:,'Vendor Name'] = max_length(raw_df,'Vendor Number', 'Vendor Name')
# Lets do the sam trick for Categories:

raw_df.loc[:,'Category'] = max_length(raw_df,'Category','Category Name')
# That's weird gotta explore
if (raw_df['Volume Sold (Liters)'].isna() == raw_df['Volume Sold (Gallons)'].isna()).all():

    raw_df.drop(columns=['Volume Sold (Gallons)'], inplace=True)
# Check if any columns can be converted to categories:

count_dict = {}

for c in tqdm(raw_df.columns):

    count_per_value = raw_df[c].value_counts()

    count_dict[c] = {

        'count_of_values': count_per_value.count(),

        'count_per_value': count_per_value

    }

    print(f"Column {c} has {count_dict[c]['count_of_values']} values.")
raw_df['Store Name'].unique()
raw_df['Store Name'].isna().sum()
raw_df['Store Number'].isna().sum()
raw_df['Address'].unique()