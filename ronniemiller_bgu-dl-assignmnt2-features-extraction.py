# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Model
from keras.layers import *
from keras.callbacks import *
from keras.regularizers import l2
from keras.optimizers import *
from keras.utils import to_categorical
import datetime
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from keras import backend as K
from sklearn.model_selection import KFold


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# define function to reduce memory usage
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
train_set = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test_set = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
history_trx = pd.read_csv("../input/historical_transactions.csv", parse_dates=['purchase_date'])
new_trx = pd.read_csv("../input/new_merchant_transactions.csv", parse_dates=['purchase_date'])
merchants_set = pd.read_csv("../input/merchants.csv")

print("shape of train : ",train_set.shape)
print("shape of test : ",test_set.shape)
print("shape of history_trx : ",history_trx.shape)
print("shape of new_trx : ",new_trx.shape)
print("shape of merchants : ",merchants_set.shape)
merchants_set.drop_duplicates(subset=['merchant_id'], keep='first', inplace=True)
# add 'year', 'month', and 'elepsed_time' features to the dataframe
for df in [train_set, test_set]:
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days # 1/2/2018 is the max date in train set
    
# create set of columns name that is numeric
numeric_col = ['elapsed_time']

# split the train set to features and target
target = train_set['target']
del train_set['target']
def get_top_merchants(trx_data):
    num_trx_per_mer = trx_data.groupby(['card_id', 'merchant_id'])['authorized_flag'].agg(['count'])
    num_trx_per_mer.reset_index(inplace=True)
    num_trx_per_mer = num_trx_per_mer.sort_values('count', ascending=False).drop_duplicates(['card_id'], keep='first')
    num_trx_per_mer = num_trx_per_mer[['card_id', 'merchant_id']]
    return num_trx_per_mer

history_top_merchants = get_top_merchants(history_trx)
new_top_merchants = get_top_merchants(new_trx)

history_top_merchants.rename(index=str, columns={"merchant_id": "history_top_merchants"}, inplace=True)
new_top_merchants.rename(index=str, columns={"merchant_id": "new_top_merchants"}, inplace=True)
train_set = pd.merge(train_set, history_top_merchants, on='card_id', how='left')
test_set = pd.merge(test_set, history_top_merchants, on='card_id', how='left')

train_set = pd.merge(train_set, new_top_merchants, on='card_id', how='left')
test_set = pd.merge(test_set, new_top_merchants, on='card_id', how='left')

train_set = pd.merge(train_set, merchants_set, left_on='history_top_merchants',right_on='merchant_id', how='left')
test_set = pd.merge(test_set, merchants_set, left_on='history_top_merchants',right_on='merchant_id', how='left')

train_set.drop(['merchant_id','history_top_merchants', 'new_top_merchants'], axis=1, inplace=True)
test_set.drop(['merchant_id','history_top_merchants', 'new_top_merchants'], axis=1, inplace=True)

del merchants_set
gc.collect()

train_set = reduce_mem_usage(train_set)
test_set = reduce_mem_usage(test_set)

del history_top_merchants
del new_top_merchants
gc.collect()
# define aggregation function on card_id that collect information from data features to crate profile to each card_id
def agg_data_trx(trx_data, col_name):
    
    trx_data['authorized_flag'] = trx_data['authorized_flag'].map({'Y':1, 'N':0})
    
    trx_data['purchase_month'] = trx_data['purchase_date'].dt.month
    
    trx_data['month_diff'] = ((datetime.datetime.today() - trx_data['purchase_date']).dt.days)//30
    trx_data['month_diff'] += trx_data['month_lag']
    
    trx_data = reduce_mem_usage(trx_data)
    
    trx_data.loc[:, 'purchase_date'] = pd.DatetimeIndex(trx_data['purchase_date']).astype(np.int64) * 1e-9
    
    agg_func = {
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['mean', 'max', 'min', 'std'],
        'month_diff': ['mean']
    }
    
    agg_data = trx_data.groupby(['card_id']).agg(agg_func)
    agg_data.columns = [col_name + '_' + '_'.join(col).strip() for col in agg_data.columns.values]
    agg_data.reset_index(inplace=True)
    
    df = (trx_data.groupby('card_id').size().reset_index(name=col_name + '_trx_count'))
    
    agg_data = pd.merge(df, agg_data, on='card_id', how='left')
    
    agg_numeric_col = [col for col in agg_data.columns if col not in ['card_id']]
    numeric_col.extend(agg_numeric_col)
    
    return agg_data

authorized_trx = history_trx[history_trx['authorized_flag'] == 'Y']
history_trx = history_trx[history_trx['authorized_flag'] == 'N']

history_trx_per_card = agg_data_trx(history_trx, 'history')
authorized_trx_per_card = agg_data_trx(authorized_trx, 'auto')
new_trx_per_card = agg_data_trx(new_trx, 'new')

# merge the new features for each card_id with the 3 basic features in train set and test set
train_set = pd.merge(train_set, history_trx_per_card, on='card_id', how='left')
test_set = pd.merge(test_set, history_trx_per_card, on='card_id', how='left')

train_set = pd.merge(train_set, authorized_trx_per_card, on='card_id', how='left')
test_set = pd.merge(test_set, authorized_trx_per_card, on='card_id', how='left')

train_set = pd.merge(train_set, new_trx_per_card, on='card_id', how='left')
test_set = pd.merge(test_set, new_trx_per_card, on='card_id', how='left')

# delete unnecessary dataframes to reduce memory usage
del history_trx_per_card
del new_trx_per_card
del authorized_trx_per_card
gc.collect()
# define aggregation function on card_id and month_lag that collect information from data features to crate profile to each card_id

def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    
    return final_group

final_group =  aggregate_per_month(authorized_trx)

train_set = pd.merge(train_set, final_group, on='card_id', how='left')
test_set = pd.merge(test_set, final_group, on='card_id', how='left')

del final_group
del authorized_trx
gc.collect()

train_set = reduce_mem_usage(train_set)
test_set = reduce_mem_usage(test_set)

train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)
target.to_csv('target.csv', index=False)