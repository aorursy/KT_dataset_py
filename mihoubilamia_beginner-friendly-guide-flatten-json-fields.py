import numpy as np

import pandas as pd

import os

import json

from pandas.io.json import json_normalize
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
CSV_PATH='/kaggle/input/ga-customer-revenue-prediction/train.csv'

JSON_COLUMNS = ['device','geoNetwork', 'totals', 'trafficSource']

def load_data(csv_path=CSV_PATH, nrows=None, json_cols=JSON_COLUMNS):

    df = pd.read_csv(csv_path, # engine='python', 

                     converters={col: json.loads for col in json_cols}, 

                     dtype={'fullVisitorId': 'str'},

                     nrows=nrows)

    for col in json_cols:

        # normalizing (flattening) each json column

        col_as_df = json_normalize(df[col])

        # renaming each column of the new dataframe that resulted form

        # normalization by concatenating its name to the name of the json column 

        # from which it was extracted so that we can keep track of the 

        # significance of the columns

        col_as_df.columns = [f"{col}.{subcol}" for subcol in col_as_df.columns]

        # replacing the original json column by the new dataframe we obtained above

        df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df

        
%%time

customer_data_flattened = load_data()
os.makedirs('tmp', exist_ok=True)

customer_data_flattened.to_feather('tmp/ga-customer-data-flattened.feather')
# to read your data in the futur uncomment the following line of code:

customer_data_flattened = pd.read_feather('tmp/ga-customer-data-flattened.feather')