# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import *
import xgboost as xgb
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set(style="darkgrid")
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from itertools import product
import time
import pickle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
# Sales table view
train.head()
# Revision of missing values
train.isnull().sum()
# Sales graph per month
sale_by_month = train.groupby('date_block_num')['item_cnt_day'].sum()
sale_by_month.plot()
#Chart of quantity of items sold
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
flierprops = dict(marker='o', markerfacecolor='red', markersize=6,
                  linestyle='none', markeredgecolor='black')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)

#Chart of item prices
plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)
#Removal of prices over 100.000
train = train[train.item_price<100000]
#Removal of items over 1.001
train = train[train.item_cnt_day<1001]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

# if subtype is nan then type
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id','type_code', 'subtype_code']]

items.drop(['item_name'], axis=1, inplace=True)
ts = time.time()
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
time.time() - ts
train['revenue'] = train['item_price'] *  train['item_cnt_day']
ts = time.time()
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float32))
time.time() - ts
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) 
time.time() - ts
ts = time.time()
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
time.time() - ts
def add_group_stats(matrix_, groupby_feats, target, enc_feat, last_periods):
    if not 'date_block_num' in groupby_feats:
        print ('date_block_num must in groupby_feats')
        return matrix_
    
    group = matrix_.groupby(groupby_feats)[target].sum().reset_index()
    max_lags = np.max(last_periods)
    for i in range(1,max_lags+1):
        shifted = group[groupby_feats+[target]].copy(deep=True)
        shifted['date_block_num'] += i
        shifted.rename({target:target+'_lag_'+str(i)},axis=1,inplace=True)
        group = group.merge(shifted, on=groupby_feats, how='left')
    group.fillna(0,inplace=True)
    for period in last_periods:
        lag_feats = [target+'_lag_'+str(lag) for lag in np.arange(1,period+1)]
        # we do not use mean and svd directly because we want to include months with sales = 0
        mean = group[lag_feats].sum(axis=1)/float(period)
        mean2 = (group[lag_feats]**2).sum(axis=1)/float(period)
        group[enc_feat+'_avg_sale_last_'+str(period)] = mean
        group[enc_feat+'_std_sale_last_'+str(period)] = (mean2 - mean**2).apply(np.sqrt)
        group[enc_feat+'_std_sale_last_'+str(period)].replace(np.inf,0,inplace=True)
        # divide by mean, this scales the features for NN
        group[enc_feat+'_avg_sale_last_'+str(period)] /= group[enc_feat+'_avg_sale_last_'+str(period)].mean()
        group[enc_feat+'_std_sale_last_'+str(period)] /= group[enc_feat+'_std_sale_last_'+str(period)].mean()
    cols = groupby_feats + [f_ for f_ in group.columns.values if f_.find('_sale_last_')>=0]
    matrix = matrix_.merge(group[cols], on=groupby_feats, how='left')
    return matrix
ts = time.time()
matrix = add_group_stats(matrix, ['date_block_num', 'item_id'], 'item_cnt_month', 'item', [6,12])
matrix = add_group_stats(matrix, ['date_block_num', 'shop_id'], 'item_cnt_month', 'shop', [6,12])
matrix = add_group_stats(matrix, ['date_block_num', 'item_category_id'], 'item_cnt_month', 'category', [12])
matrix = add_group_stats(matrix, ['date_block_num', 'city_code'], 'item_cnt_month', 'city', [12])
matrix = add_group_stats(matrix, ['date_block_num', 'type_code'], 'item_cnt_month', 'type', [12])
matrix = add_group_stats(matrix, ['date_block_num', 'subtype_code'], 'item_cnt_month', 'subtype', [12])
time.time() - ts
# First use the target coding of each group, then change the month to create lags functions
def target_encoding(matrix_, groupby_feats, target, enc_feat, lags):
    print ('target encoding for',groupby_feats)
    group = matrix_.groupby(groupby_feats).agg({target:'mean'})
    group.columns = [enc_feat]
    group.reset_index(inplace=True)
    matrix = matrix_.merge(group, on=groupby_feats, how='left')
    matrix[enc_feat] = matrix[enc_feat].astype(np.float16)
    matrix = lag_feature(matrix, lags, enc_feat)
    matrix.drop(enc_feat, axis=1, inplace=True)
    return matrix
ts = time.time()
matrix = target_encoding(matrix, ['date_block_num'], 'item_cnt_month', 'date_avg_item_cnt', [1])
matrix = target_encoding(matrix, ['date_block_num', 'item_id'], 'item_cnt_month', 'date_item_avg_item_cnt', [1,2,3,6,12])
matrix = target_encoding(matrix, ['date_block_num', 'shop_id'], 'item_cnt_month', 'date_shop_avg_item_cnt', [1,2,3,6,12])
matrix = target_encoding(matrix, ['date_block_num', 'item_category_id'], 'item_cnt_month', 'date_cat_avg_item_cnt', [1])
matrix = target_encoding(matrix, ['date_block_num', 'shop_id', 'item_category_id'], 'item_cnt_month', 'date_shop_cat_avg_item_cnt', [1])
matrix = target_encoding(matrix, ['date_block_num', 'city_code'], 'item_cnt_month', 'date_city_avg_item_cnt', [1])
matrix = target_encoding(matrix, ['date_block_num', 'item_id', 'city_code'], 'item_cnt_month', 'date_item_city_avg_item_cnt', [1])
time.time() - ts
ts = time.time()
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

time.time() - ts
ts = time.time()
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
time.time() - ts
matrix['month'] = matrix['date_block_num'] % 12
matrix['year'] = (matrix['date_block_num'] / 12).astype(np.int8)
#Month since the last sale of each pair of shops/article.
ts = time.time()
last_sale = pd.DataFrame()
for month in range(1,35):    
    last_month = matrix.loc[(matrix['date_block_num']<month)&(matrix['item_cnt_month']>0)].groupby(['item_id','shop_id'])['date_block_num'].max()
    df = pd.DataFrame({'date_block_num':np.ones([last_month.shape[0],])*month,
                       'item_id': last_month.index.get_level_values(0).values,
                       'shop_id': last_month.index.get_level_values(1).values,
                       'item_shop_last_sale': last_month.values})
    last_sale = last_sale.append(df)
last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int8)

matrix = matrix.merge(last_sale, on=['date_block_num','item_id','shop_id'], how='left')
time.time() - ts
#Month since the last sale of each item.
ts = time.time()
last_sale = pd.DataFrame()
for month in range(1,35):    
    last_month = matrix.loc[(matrix['date_block_num']<month)&(matrix['item_cnt_month']>0)].groupby('item_id')['date_block_num'].max()
    df = pd.DataFrame({'date_block_num':np.ones([last_month.shape[0],])*month,
                       'item_id': last_month.index.values,
                       'item_last_sale': last_month.values})
    last_sale = last_sale.append(df)
last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int8)

matrix = matrix.merge(last_sale, on=['date_block_num','item_id'], how='left')
time.time() - ts
# Months from the first sale for each pair of shops/item and for item only.
ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts
matrix = matrix[matrix.date_block_num > 11]
matrix.columns
data = matrix[[
    'date_block_num',
    'shop_id',
    #'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code','subtype_code',
    'item_cnt_month_lag_1','item_cnt_month_lag_2','item_cnt_month_lag_3','item_cnt_month_lag_6','item_cnt_month_lag_12',
    'item_avg_sale_last_6', 'item_std_sale_last_6',
    'item_avg_sale_last_12', 'item_std_sale_last_12',
    'shop_avg_sale_last_6', 'shop_std_sale_last_6',
    'shop_avg_sale_last_12', 'shop_std_sale_last_12',
    'category_avg_sale_last_12', 'category_std_sale_last_12',
    'city_avg_sale_last_12', 'city_std_sale_last_12',
    'type_avg_sale_last_12', 'type_std_sale_last_12',
    'subtype_avg_sale_last_12', 'subtype_std_sale_last_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_2','date_item_avg_item_cnt_lag_3','date_item_avg_item_cnt_lag_6','date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1','date_shop_avg_item_cnt_lag_2','date_shop_avg_item_cnt_lag_3','date_shop_avg_item_cnt_lag_6','date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month','year',
    'item_shop_last_sale','item_last_sale',
    'item_shop_first_sale','item_first_sale',
]]

cat_feats = ['shop_id','city_code','item_category_id','type_code','subtype_code']
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
ts = time.time()

model_lgbm = LGBMRegressor(
    max_depth = 8,
    n_estimators = 500,
    colsample_bytree=0.7,
    min_child_weight = 300,
    reg_alpha = 0.1,
    reg_lambda = 1,
    random_state = 42,
)

model_lgbm.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=10, 
    early_stopping_rounds = 40,
    categorical_feature = cat_feats) # use LGBM's build-in categroical features.

time.time() - ts
Y_pred_lgbm = model_lgbm.predict(X_valid).clip(0, 20)
Y_test_lgbm = model_lgbm.predict(X_test).clip(0, 20)

X_train_lgbm = pd.DataFrame({
    "ID": np.arange(Y_pred_lgbm.shape[0]), 
    "item_cnt_month": Y_pred_lgbm
})
X_train_lgbm.to_csv('lgb_valid.csv', index=False)

submission_lgbm = pd.DataFrame({
    "ID": np.arange(Y_test_lgbm.shape[0]), 
    "item_cnt_month": Y_test_lgbm
})
submission_lgbm.to_csv('submission_lgb.csv', index=False)
ts = time.time()

model_xgb = XGBRegressor(
    max_depth=7,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    gamma = 0.005,
    eta=0.1,    
    seed=42)

model_xgb.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=10, 
    early_stopping_rounds = 40,
    )

time.time() - ts
Y_pred_xgb = model_xgb.predict(X_valid).clip(0, 20)
Y_test_xgb = model_xgb.predict(X_test).clip(0, 20)

X_train_xgb = pd.DataFrame({
    "ID": np.arange(Y_pred_xgb.shape[0]), 
    "item_cnt_month": Y_pred_xgb
})
X_train_xgb.to_csv('xgb_valid.csv', index=False)

submission_xgb = pd.DataFrame({
    "ID": np.arange(Y_test_xgb.shape[0]), 
    "item_cnt_month": Y_test_xgb
})
submission_xgb.to_csv('submission_xgb.csv', index=False)
pkl_filename = "pickle_model_LGBM.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_lgbm, file)
pkl_filename = "pickle_model_XGB.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_xgb, file)
pkl_filename = "data.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(data, file)