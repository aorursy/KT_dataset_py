import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize

import os
print(os.listdir("../input"))
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
%%time
df_train = load_df()
df_test = load_df("../input/test.csv")
constant_columns = [col for col in df_train.columns if df_train[col].nunique(dropna=False)==1]

df_train.drop(columns=constant_columns,inplace=True)
df_test.drop(columns=constant_columns,inplace=True)
%%time
df_train.to_csv("df_train.csv", index=False)
df_test.to_csv("df_test.csv", index=False)