# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/elo-merchant-category-recommendation/"))
# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import gc
gc.enable()
gc.collect()
#Reduce the memory usage, from "Elo Merchant Category Recommendation by [team Tour_de_Force]"
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
train = reduce_mem_usage(pd.read_csv('../input/elo-merchant-category-recommendation/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/elo-merchant-category-recommendation/test.csv'))
merchants = reduce_mem_usage(pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv'))
new_trans = reduce_mem_usage(pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv'))
historical_trans = reduce_mem_usage(pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv'))
#historical_trans.head()
#new_trans.head()
one_hot_cols = ['category_1','category_2', 'category_3',]
bool_cols= ['authorized_flag']
numerical_cols = ['purchase_amount', 'installments', 'month_lag',] #'purchase_date',  <-- this has to be processed 
categorical_cols = ['city_id', 'state_id', 'merchant_category_id','subsector_id',]
merchant_id = ['merchant_id']
# purchase_date cleaning and conversion TBD
for c in bool_cols:
    new_trans[c] = new_trans[c].apply(lambda x: True if x=='Y' else False).astype(bool)
    historical_trans[c] = historical_trans[c].apply(lambda x: True if x=='Y' else False).astype(bool)

#print(historical_trans[bool_cols].describe())
#print()
#print(new_trans[bool_cols].describe())
    
for c in categorical_cols:
    historical_trans[c] = historical_trans[c].astype('category')
    new_trans[c] = new_trans[c].astype('category')
for c in numerical_cols:
    print(historical_trans[c].describe())
    print(new_trans[c].describe())
    print(sum(pd.isna(historical_trans[c])), sum(pd.isna(new_trans[c])))
for c in one_hot_cols:
    historical_trans[c] = historical_trans[c].astype('category')
    new_trans[c] = new_trans[c].astype('category')
    '''
    print('column name: ', c)
    print(historical_trans[c].value_counts())
    print(sum(pd.isna(historical_trans[c])))
    print()
    print(new_trans[c].value_counts())
    print(sum(pd.isna(new_trans[c])))
    print()
    '''
one_hot_new = pd.get_dummies(new_trans[one_hot_cols], dummy_na=True)
one_hot_hist = pd.get_dummies(historical_trans[one_hot_cols], dummy_na=True)
historical_trans = pd.concat([historical_trans, one_hot_hist], axis = 1)
new_trans = pd.concat([new_trans, one_hot_hist], axis =1)
agg_function = {
    'purchase_amount' : ['mean', 'sum', 'std', 'nunique', 'max', 'min'],
    'installments': ['mean', 'sum', 'std', 'nunique', 'max', 'min'],
    'month_lag' : ['mean', 'std', 'nunique', 'max', 'min'],
    
    'city_id': ['count', 'nunique'],
    'state_id': ['count', 'nunique'],
    'merchant_category_id': ['count', 'nunique'],
    'subsector_id': ['count', 'nunique'],
    
   
    'authorized_flag': ['sum', 'count'],   
    'category_1_N': ['mean', 'sum'],
    'category_1_Y': ['mean', 'sum'], 
    'category_1_nan': ['mean', 'sum'],
    'category_2_1.0': ['mean', 'sum'], 
    'category_2_2.0': ['mean', 'sum'], 
    'category_2_3.0': ['mean', 'sum'],
    'category_2_4.0': ['mean', 'sum'], 
    'category_2_5.0': ['mean', 'sum'], 
    'category_2_nan': ['mean', 'sum'],
    'category_3_A': ['mean', 'sum'], 
    'category_3_B': ['mean', 'sum'], 
    'category_3_C': ['mean', 'sum'], 
    'category_3_nan': ['mean', 'sum'],
}
new_trans_by_card_id = new_trans.groupby(['card_id']).agg(agg_function)
historical_trans_by_card_id = historical_trans.groupby(['card_id']).agg(agg_function)
trans_by_card = historical_trans_by_card_id.merge(
    new_trans_by_card_id,how='left', left_index=True, right_index=True).reset_index()
l_org = trans_by_card.columns.values
l = [p[0] + '_' + p[1] for p in l_org]
l[0] = 'card_id'
trans_by_card.columns = l
trans_by_card.head()
del [historical_trans, new_trans ]
gc.collect()
trans_by_card.to_csv('trans_by_card.csv')
#np.log(historical_trans[historical_trans['authorized_flag']=='Y'].card_id.value_counts()).hist()
trans_by_card.head()
df = pd.read_csv('../input/trans_by_card.csv')
df.head()
