import numpy as np 
import pandas as pd
import json
from pandas.io.json import json_normalize
import os
import gc
print(os.listdir("../input"))
def load_data(train_url, test_url):
    """
    load train and test data 
    """
    dtype={
            'channelGrouping': str,
            'geoNetwork': str,
            'date': str,
            'fullVisitorId': str,
            'sessionId': str,
            'totals': str,
            'device': str
        }
    
    df_train = pd.read_csv(train_url, dtype=dtype)
    df_test =  pd.read_csv(test_url, dtype=dtype)

    df_train['is_train'] = 1
    df_test['is_train'] = 0

    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    del df_train; del df_test; gc.collect()
    return df

def normalize_json_cols(df_t, json_cols):
    """
    returns df with columns for keys in a column that contains a json string
    """
    df = df_t.copy()
    for i in json_cols:
        temp = pd.io.json.json_normalize(df[i].apply(json.loads))
        temp.columns = [i + '_' + j for j in temp.columns]
        df = pd.concat([df, temp], axis=1)
    df.drop(json_cols, axis=1, inplace=True)
    return df

def remove_junk_data(df, dont_del=[]):
    """
    remove columns with only 1 unique value
    """
    unique_counts = df.nunique()
    unique_counts.drop(dont_del, inplace=True)
    return df.drop(list(unique_counts.index[unique_counts == 1]), axis=1)

def coerce_numeric_columns(df, numerics):
    """
    force certain numeric columns to be... numeric
    """
    df[numerics] = df[numerics].apply(pd.to_numeric, errors='coerce')
    return df

def run():
    """
    return the cleaned up dataframe
    """
    df = load_data('../input/train.csv', '../input/test.csv')
    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
    dont_del = ['totals_newVisits', 'totals_bounces']
    df = normalize_json_cols(df, json_cols)
    df = remove_junk_data(df, dont_del=dont_del)
    numerics = ['totals_hits', 'totals_pageviews', 'totals_transactionRevenue', 'totals_newVisits', 'totals_bounces']
    df = coerce_numeric_columns(df, numerics)
    return df
df = run()
df.to_csv('starting_point.csv', index=False)
