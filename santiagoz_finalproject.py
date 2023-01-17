# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import gc

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
DATA_FOLDER = '../input/competitive-data-science-predict-future-sales'
items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_cats = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
train = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
submission = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))
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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
print("Items")
print(items.head(2))
print("\nItem Catageries")
print(item_cats.tail(2))
print("\nShops")
print(shops.sample(n=2))
print("\nTraining Data Set")
print(train.sample(n=3,random_state=1))
print("\nTest Data Set")
print(test.sample(n=3,random_state=1))
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)
train = train[train.item_price<90001]
train = train[train.item_cnt_day<901]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median
import math

grouped_shop_id = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5,ncols=2,sharex=False,sharey=False,figsize=(16,20))
count = 0
num_graph = 10
id_per_graph = math.ceil(grouped_shop_id.shop_id.max()/num_graph)

for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped_shop_id[np.logical_and(count*id_per_graph <= grouped_shop_id['shop_id'], grouped_shop_id['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1
del grouped_shop_id
train_join_item = pd.merge(train, items, how='left', on=['item_id'])
train_join_item = train_join_item.drop('item_name', axis=1) # no need to use column item_name
train_join_item.head(10)
grouped_item_category_id = pd.DataFrame(train_join_item.groupby(['item_category_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5,ncols=2,sharex=False,sharey=False,figsize=(16,20))
count = 0
num_graph = 10
id_per_graph = math.ceil(grouped_item_category_id.item_category_id.max()/num_graph)

for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='item_category_id', data=grouped_item_category_id[np.logical_and(count*id_per_graph <= grouped_item_category_id['item_category_id'], grouped_item_category_id['item_category_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1
del grouped_item_category_id
from itertools import product

# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Groupby data to get shop-item-month aggregates
gb = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum','item_price':np.mean}).reset_index()

# Join it to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)
all_data = pd.merge(all_data, items, how='left', on=['item_id'])
all_data = all_data.drop('item_name', axis=1) # no need to use column item_name

print(all_data.head())

for type_id in ['item_id', 'shop_id', 'item_category_id']:
    for column_id, aggregator, aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:
        
        gb = train_join_item.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        gb.columns = [type_id+'_'+column_id+'_'+aggtype,type_id,'date_block_num']
        
        all_data = pd.merge(all_data, gb, on=['date_block_num',type_id], how='left')

all_data
del grid
del gb
del cur_shops
del cur_items
reduce_mem_usage(all_data, verbose=True)
lag_variables  = list(all_data.columns[6:])+['item_cnt_day']
lags = [1, 2, 3, 6]
for lag in lags:
    sales_new_df = all_data.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    all_data = pd.merge(all_data, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
    
all_data
del sales_new_df
for feat in all_data.columns:
    if 'item_cnt' in feat:
        all_data[feat]=all_data[feat].fillna(0)
    elif 'item_price' in feat:
        all_data[feat]=all_data[feat].fillna(all_data[feat].median())
cols_to_drop = lag_variables[:-1] + ['item_price']
training = all_data.drop(cols_to_drop,axis=1)
test.head()
test['date_block_num'] = 34
test = pd.merge(test, items, on='item_id', how='left')
for lag in lags:

    sales_new_df = all_data.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    test = pd.merge(test, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
test = test.drop(['ID', 'item_name'], axis=1)
test_columns = test.columns
training_columns = set(training.drop('item_cnt_day',axis=1).columns)
print(len(test_columns))
print(len(training_columns))
for i in test_columns:
    assert i in training_columns
for i in training_columns:
    assert i in test_columns
for feat in test.columns:
    if 'item_cnt' in feat:
        test[feat]=test[feat].fillna(0)
    elif 'item_price' in feat:
        test[feat]=test[feat].fillna(test[feat].median())
test[['shop_id','item_id']+['item_cnt_day_lag_'+str(x) for x in [1,2,3]]].head()
print(training[training['shop_id'] == 5][training['item_id'] == 5233][training['date_block_num'] == 33]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5233][training['date_block_num'] == 32]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5233][training['date_block_num'] == 31]['item_cnt_day'])
X_train = training[training.date_block_num < 33].drop(['item_cnt_day'], axis=1)
Y_train = training[training.date_block_num < 33]['item_cnt_day']
X_valid = training[training.date_block_num == 33].drop(['item_cnt_day'], axis=1)
Y_valid = training[training.date_block_num == 33]['item_cnt_day']
X_test = test
#X_train.to_csv('X_train.csv', index=False)
#Y_train.to_csv('Y_train.csv', index=False)
#X_valid.to_csv('X_valid.csv', index=False)
#Y_valid.to_csv('Y_valid.csv', index=False)
#X_test.to_csv('X_test.csv', index=False)
del training
del test
gc.collect()
from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb

xgbtrain = xgb.DMatrix(X_train, Y_train)

param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'} # random parameters
model = xgb.train(param, xgbtrain)
x=plot_importance(model)
x.figure.set_size_inches(10, 30) 
score = model.get_score(importance_type='weight')

# list out keys and values separately 
key_list = list(score.keys()) 
val_list = list(score.values())

top_score = list(filter(lambda x: x >= 200, val_list))

top_feat = []

for i in top_score:
    feat = key_list[val_list.index(i)]
    top_feat += [feat]
    
print(top_feat)
xgbpredict = xgb.DMatrix(X_test)
pred_xgb = model.predict(xgbpredict).clip(0, 20)
del model
del xgbpredict
gc.collect()
pd.Series(pred_xgb).describe()
sub_df_xgb = pd.DataFrame({'ID':X_test.index,'item_cnt_month': pred_xgb })
sub_df_xgb.head()
sub_df_xgb.to_csv('submission_xgb.csv',index=False)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)
pred_lr = lr.predict(X_test).clip(0, 20)
pd.Series(pred_lr).describe()
sub_df_lr = pd.DataFrame({'ID':X_test.index,'item_cnt_month': pred_lr })
sub_df_lr.head()
sub_df_lr.to_csv('submission_lr.csv',index=False)
import lightgbm as lgb

lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':1, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0 
              }
model_lgb = lgb.train(lgb_params, lgb.Dataset(X_train, label=Y_train), 100)
pred_lgb = model_lgb.predict(X_test).clip(0, 20)
pd.Series(pred_lgb).describe()
sub_df_lgb = pd.DataFrame({'ID':X_test.index,'item_cnt_month': pred_lgb })
sub_df_lgb.head()
sub_df_lgb.to_csv('submission_lgb.csv',index=False)
pred_w_avg = 0.7*pred_xgb + 0.15*pred_lr + 0.15*pred_lgb
pd.Series(pred_w_avg).describe()
sub_df_w_avg = pd.DataFrame({'ID':X_test.index,'item_cnt_month': pred_w_avg })
sub_df_w_avg.head()
sub_df_w_avg.to_csv('submission_w_avg.csv',index=False)