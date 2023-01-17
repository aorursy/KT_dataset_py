import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
train_df = load_df()
test_df = load_df("../input/test.csv")
train_df.to_csv('tr.csv.gz',index=False,compression='gzip')
test_df.to_csv('te.csv.gz',index=False,compression='gzip')