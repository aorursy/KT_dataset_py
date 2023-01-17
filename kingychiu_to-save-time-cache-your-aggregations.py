# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import time
# Any results you write to the current directory are saved as output.
def query_cached_dataframe(key, force_update=False):
    path = '{}.pickle'.format(key)
    # check if a file exist
    exists = os.path.isfile(path)
    if exists and not force_update:
        print('cache hits', key)
        return (True, pd.read_pickle(path))
    else:
        print('cache misses', key)

        def do_cache(df):
            df.to_pickle(path)

        return (False, do_cache)
df = pd.read_csv('../input/new_merchant_transactions.csv')
def run():
    by = 'card_id'
    feature = 'purchase_amount'
    agg_dfs = []
    for agg in ['sum', 'mean', 'var', 'min', 'max']:
        new_feature_name = '{}_{}_{}'.format(feature, agg, by)
        cache_hit, cache_result = query_cached_dataframe(new_feature_name)
        if cache_hit:
            agg_df = cache_result
        else:
            agg_df = df.groupby(by)[feature].agg(agg).reset_index()\
                .rename(columns={feature: new_feature_name})
            cache_result(agg_df)
        agg_dfs.append(agg_df)
    return agg_dfs
start = time.time()
agg_dfs = run()
print(time.time() - start)
display(agg_dfs[0].head())
start = time.time()
agg_dfs = run()
print(time.time() - start)
display(agg_dfs[0].head())