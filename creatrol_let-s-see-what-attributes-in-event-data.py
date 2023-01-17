import numpy as np

import pandas as pd

import json

import gc

import ast

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('max_columns', 100)

pd.set_option('max_rows', 100)
%%time

train_csv = '/kaggle/input/data-science-bowl-2019/train.csv'

test_csv = '/kaggle/input/data-science-bowl-2019/test.csv'

specs_csv = '/kaggle/input/data-science-bowl-2019/specs.csv'



all_train_event_attributes, all_test_event_attributes = [], []

train_count, test_count = 0, 0

for chunk in pd.read_csv(train_csv,chunksize=10000):

    chunk_attributes = chunk['event_data'].apply(lambda x: list(json.loads(x).keys()))

    all_train_event_attributes.extend([y for x in chunk_attributes.to_list() for y in x])

    train_count += chunk.shape[0]

    

for chunk in pd.read_csv(test_csv,chunksize=10000):

    chunk_attributes = chunk['event_data'].apply(lambda x: list(json.loads(x).keys()))

    all_test_event_attributes.extend([y for x in chunk_attributes.to_list() for y in x])

    test_count += chunk.shape[0]
%%time

count_train = Counter(all_train_event_attributes)

count_test = Counter(all_test_event_attributes)



def get_count_df(count_dict, total):

    df = pd.DataFrame.from_dict(count_dict, orient='index')

    df['attribute']=df.index

    df.columns = ['count', 'attribute']

    df.sort_values(by=['count'], axis=0, ascending=False, inplace=True)

    df['pct'] = df['count'] / total

    return df



count_train_df = get_count_df(count_train, train_count)

count_test_df = get_count_df(count_test, test_count)
plt.figure(figsize=(10, 30))

sns.set(style='whitegrid')

ax = sns.barplot(x='pct', y='attribute', data=count_train_df.head(50))
plt.figure(figsize=(10, 30))

sns.set(style='whitegrid')

ax = sns.barplot(x='pct', y='attribute', data=count_test_df.head(50))
specs = pd.read_csv(specs_csv)

specs_parse = lambda _col:str([x['name'] for x in json.loads(_col)])

specs['attribute_list'] = specs['args'].apply(lambda _col: specs_parse(_col))

specs.head()