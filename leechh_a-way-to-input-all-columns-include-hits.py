import gc
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

import os
print(os.listdir("../input"))

def todict(dic, key, value):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]
    return dic


def resolve_json(hitsdic, hits_json, key='NoneName'):
    if type(hits_json) == list:
        if len(hits_json) == 0:
            pass
        else:
            for subjson in hits_json:
                hitsdic = resolve_json(hitsdic, subjson)
    elif type(hits_json) == dict:
        for i in hits_json.keys():
            hitsdic = resolve_json(hitsdic, hits_json[i],i)
    else:
        hitsdic = todict(hitsdic, key, hits_json)
    return hitsdic


def complex_replace(x):
    dic = {}
    return resolve_json(dic, json.loads(x.replace('\'','\"'). \
                                        replace('TRUE','true'). \
                                        replace('True','true'). \
                                        replace('FALSE','false'). \
                                        replace('False','false'). \
                                        replace(', \"',', !&~'). \
                                        replace('\", ','!&~, '). \
                                        replace('\": ','!&~: '). \
                                        replace(': \"',': !&~'). \
                                        replace(' {\"',' {!&~'). \
                                        replace('\"}, ','!&~}, '). \
                                        replace('[{\"','[{!&~'). \
                                        replace('\"}]','!&~}]'). \
                                        replace('\"','_'). \
                                        replace('!&~','\"'). \
                                        encode('gbk','ignore'). \
                                        decode('utf-8','ignore'). \
                                        replace('\\','')))


def replace(x):
    return  json.loads(x)


def load_df(csv_path, nrows=None, chunksize=10_000, percent=100):
    n=1
    df_list = []
    feature = ['device', 'hits', 'customDimensions', 'geoNetwork', 'totals', 'trafficSource']
    chunk = pd.read_csv(csv_path,
                        nrows=nrows, 
                        chunksize=chunksize, 
                        dtype={'fullVisitorId': 'str'}) # Important!!
    for subchunk in chunk:
        for column in feature:
            if column in ['customDimensions','hits']:
                column_as_df = json_normalize(subchunk[column].apply(complex_replace))
            else:
                column_as_df = json_normalize(subchunk[column].apply(replace))
            column_as_df.columns = [f'{column}_{subcolumn}' for subcolumn in column_as_df.columns]
            subchunk.drop(column, axis=1, inplace=True)
            subchunk = subchunk.reset_index(drop=True).merge(column_as_df,
                                           right_index=True,
                                           left_index=True)
        n = n+1
        df_list.append(subchunk.astype('str'))
        del column_as_df, subchunk
    return pd.concat(df_list, ignore_index=True, sort=True)

# If you want to load all the data, change 1_000 to None and change chunksize.
train = load_df('../input/train_v2.csv',nrows=1_000, chunksize=100)
train.head()