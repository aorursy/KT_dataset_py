import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import product

from tqdm import tqdm, tqdm_notebook



import lightgbm as lgb

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



import tensorflow as tf

import keras

import pickle

import gc

import os
def downcast_dtypes(df):    

    # Select columns to downcast

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    

    return df
print(os.listdir("../input"))
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

items_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

test_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

sales
items
items_categories
sample_submission
test_data
test_block = sales['date_block_num'].max() + 1

test_data['date_block_num'] = test_block

test_data = test_data.drop(columns=['ID'])

test_data.head()
# Create new columns

index_cols = ['shop_id', 'item_id', 'date_block_num']



# For every month we create a grid from all shops/items combinations from that month

grid = []

for block_num in sales['date_block_num'].unique():

    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()

    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)

grid = pd.concat([grid, test_data])
grid
# Groupby data to get shop-item-month aggregates

gb = sales.groupby(index_cols, as_index=False)['item_cnt_day'].sum()

gb = gb.rename(columns={'item_cnt_day': 'target'})

all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)



# Same as above but with shop-month aggregates

gb = sales.groupby(['shop_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()

gb = gb.rename(columns={'item_cnt_day': 'target_shop'})

all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)



# Same as above but with item-month aggregates

gb = sales.groupby(['item_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()

gb = gb.rename(columns={'item_cnt_day': 'target_item'})

all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)



# Downcast dtypes from 64 to 32 bit to save memory

all_data = downcast_dtypes(all_data)

del grid, gb 

gc.collect();
all_data
# List of columns that we will use to create lags

cols_to_rename = list(all_data.columns.difference(index_cols))

shift_range = [1, 2, 3, 4, 5, 12]



for month_shift in tqdm_notebook(shift_range):

    train_shift = all_data[index_cols + cols_to_rename].copy()

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift



    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x

    train_shift = train_shift.rename(columns=foo)



    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

    

del train_shift
all_data.head()

# Don't use old data from year 2013

all_data = all_data[all_data['date_block_num'] >= 12]



# List of all lagged features

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]

# We will drop these at fitting stage

to_drop_cols = ['target_item', 'target_shop', 'target', 'date_block_num']

to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num']



# Category for each item

item_category_mapping = items[['item_id', 'item_category_id']].drop_duplicates()

all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')

all_data = downcast_dtypes(all_data)



gc.collect()
to_drop_cols
all_data.head()

dates = all_data['date_block_num']



dates_train  = dates[dates <  test_block]

dates_test  = dates[dates == test_block]
X_train = all_data.loc[dates <  test_block].drop(to_drop_cols, axis=1)

X_test =  all_data.loc[dates == test_block].drop(to_drop_cols, axis=1)



y_train = all_data.loc[dates <  test_block, 'target'].values

y_test =  all_data.loc[dates == test_block, 'target'].values
X_train
y_train
target_range = [0, 20]

target_range
lgb_params = {

               'feature_fraction': 0.75,

               'metric': 'rmse',

               'nthread':1, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.7, 

               'learning_rate': 0.04, 

               'objective': 'mse', 

               'bagging_seed': 2**7,

               'num_leaves': 2**7,

               'bagging_freq':1,

               'verbose':0 

              }



model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 500)

pred_lgb = model.predict(X_test).clip(*target_range)
pred_lgb

submission = pd.DataFrame({'ID': sample_submission.ID, 'item_cnt_month': pred_lgb})

submission.to_csv('submission.csv', index=False)