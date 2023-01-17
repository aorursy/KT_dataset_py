import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import gc
def zero_entropy(data):
    const_cols = [c for c in data.columns if data[c].nunique(dropna=False) == 1]
    if len(const_cols) > 0:
        print("The following columns will be dropped since they have only one value: \n")
        print(const_cols)
        for col in const_cols:
            del data[col]
    return data


def na_dropper(data):
    tot = data.shape[0]
    for col in data.columns:
        mis = data[col].isna().sum()
        if ((mis/tot) > 0.7) and ('transactionRevenue' not in col): # quick escape from making a mistake
            print("The column {} will be dropped because more than 70% of the entries are missing".format(col))
            del data[col]
    return data

            
def light_import(data_path):
    # first, simple columns
    simple_cols = ['channelGrouping', 'fullVisitorId', 'sessionId', 
              'visitId', 'visitNumber', 'visitStartTime']
    result = pd.read_csv(data_path, usecols=simple_cols, dtype={'fullVisitorId': 'str'})
    # cleaning useless columns
    result = zero_entropy(result)
    result = na_dropper(result)
    # then focus on the complex column
    complex_cols = ['geoNetwork', 'device', 'totals', 'trafficSource']
    for col in complex_cols:
        print("Importing {}...".format(col)) # to watch something happening
        tmp = pd.read_csv(data_path, usecols=[col])
        tmp = json_normalize(tmp[col].apply(json.loads))
        tmp.columns = [f"{col}_{subcolumn}" for subcolumn in tmp.columns]
        # cleaning columns
        tmp = zero_entropy(tmp)
        tmp = na_dropper(tmp)
        # mergin what is left
        result = result.merge(tmp, left_index=True, right_index=True)
        # remove the garbage
        del tmp
        gc.collect()
    return result
%%time
df_train = light_import('../input/train.csv')
%%time
df_test = light_import('../input/test.csv')
df_train.columns
