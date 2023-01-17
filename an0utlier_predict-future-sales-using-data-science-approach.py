# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import LabelEncoder


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle
sys.version_info
plt.style.use('ggplot')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
# set index to ID to avoid droping it later
test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
train.head()
fig, ax = plt.subplots(nrows= 1, ncols= 2, figsize = (12,4))

sns.distplot(train['item_price'], kde= False, ax= ax[0])
sns.distplot(train['item_cnt_day'], kde= False, ax= ax[1])
fig, ax = plt.subplots(nrows= 1, ncols= 2, figsize = (12,4))

sns.boxplot(x = train['item_price'],ax= ax[0])
sns.boxplot(x = train['item_cnt_day'],ax= ax[1])
train[['item_price', 'item_cnt_day']].describe().transpose()
train['item_price'].nlargest(5)
train['item_cnt_day'].nlargest(5)
train = train[train['item_price'] < 100000]
train = train[train['item_cnt_day'] < 1001]
train['item_price'].nsmallest(5)
train[train['item_price'] <0]
median_value = train[(train['shop_id'] == 32) & (train['item_id'] == 2973) & (train['item_price'] > 0)]['item_price'].median()

train.loc[train.item_price<0, 'item_price'] = median_value
shops.head()
shop_names = list(shops['shop_name'])
print(f"Unique Number of Shop Names : {shops['shop_name'].nunique()}")
print(f"Unique Number of Shop IDs : {shops['shop_id'].nunique()}")
shops['city_name'] = shops['shop_name'].apply(lambda x : x.split()[0])
shops.head()
shops.drop('shop_name', axis= 1, inplace= True)
shops.head()
label_encoder = LabelEncoder()
shops['city_code'] = label_encoder.fit_transform(shops['city_name'])
shops.drop('city_name', axis= 1, inplace= True)
shops.head()
items.head()
#Creating Item type & subtype features
cats['item_type'] = cats['item_category_name'].apply(lambda x : x.split('-')[0].strip())
cats['item_subtype'] = cats['item_category_name'].apply(lambda x : x.split('-')[1].strip() if len(x.split('-')) > 1 else x.split('-')[0].strip())

#Encoding them
cats['item_type_code'] = LabelEncoder().fit_transform(cats['item_type'])
cats['item_subtype_code'] = LabelEncoder().fit_transform(cats['item_subtype'])
cats.drop(['item_category_name', 'item_type', 'item_subtype'], axis= 1, inplace= True)
cats.head()
items = items.merge(cats, on= 'item_category_id', how= 'left')
items.drop('item_name', inplace= True, axis= 1)
items.head()
test.head()
#Items which are present in Test but not in Train set
len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id))))
print(f"Unique Items in Test Set: {test['item_id'].nunique()}")
print(f"Unique Shops in Test Set: {test['shop_id'].nunique()}")
print(f"Total Shop Item pairs : {len(test)}")
train['date_block_num'].unique()
matrix = []

cols = ['date_block_num', 'shop_id', 'item_id']

for i in range(34) :
    sales = train[train['date_block_num'] == i]
    matrix.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique() )), dtype='int16'))
matrix = pd.DataFrame(np.vstack(matrix), columns= cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)

matrix.head()
group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day' : 'sum'})
group.columns = ['item_cnt_month']
group.reset_index(inplace = True)

group.head()
matrix = pd.merge(matrix, group, on= cols, how ='left')

matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16)

matrix.head()
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

test.head()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) 

matrix.head()
#pulling shop features
matrix = pd.merge(matrix, shops, on= 'shop_id', how= 'left')
matrix = pd.merge(matrix, items, on= 'item_id', how= 'left')

matrix.head()
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['item_type_code'] = matrix['item_type_code'].astype(np.int8)
matrix['item_subtype_code'] = matrix['item_subtype_code'].astype(np.int8)

matrix.head()
lags = [1,2,3,6,12]

for lag in lags :
    matrix[f'Previous_{lag}_month_sales'] = matrix.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)
    
matrix.fillna(0, inplace= True)
matrix.head()
def mean_encoded_features(matrix, columns, lags) :
    
    group = matrix.groupby(columns).agg({'item_cnt_month' : 'mean'})
    new_column_name = '_'.join(columns) + '_item_cnt'
    group.columns = [new_column_name] 
    group.reset_index(inplace = True)
    
    matrix = pd.merge(matrix, group, on= columns, how= 'left')
    matrix[new_column_name] = matrix[new_column_name].astype(np.float16)
    
    for lag in lags :
        matrix[f'lag_{lag + 1}_{new_column_name}'] = matrix.groupby(['shop_id', 'item_id'])[new_column_name].shift(lag)
        
    matrix.drop(new_column_name, axis=1, inplace=True)
        
    return matrix
        
#Month Sales Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num'], lags= [1])

#Month Item Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'item_id'], lags= [1,2,3,6,12])

#Month Shop Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'shop_id'], lags= [1,2,3,6,12])

#Month Item Category Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'item_category_id'], lags= [1,2,3,6,12])

#Month Shop Category Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'shop_id', 'item_category_id'], lags= [1])

#Month Shop Type Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'shop_id', 'item_type_code'], lags= [1])

#Month Shop SubType Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'shop_id', 'item_subtype_code'], lags= [1])

#Month City Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'city_code'], lags= [1])

#Month Item City Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'item_id', 'city_code'], lags= [1])

#Month Type Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'item_type_code'], lags= [1])

#Month Sub-Type Level
matrix = mean_encoded_features(matrix, columns= ['date_block_num', 'item_subtype_code'], lags= [1])

#Adding avg price of each item
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_price'] = matrix['item_avg_price'].astype(np.float16)
#Adding Average Price every month of each item
group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_price'] = matrix['date_item_avg_price'].astype(np.float16)
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
train['revenue'] = train['item_price'] *  train['item_cnt_day']

#Month & Shop wise revenue
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

#Shop wise Revenue
group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = mean_encoded_features(matrix, columns= ['delta_revenue'], lags= [1])

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
#month as per calender
matrix['month'] = matrix['date_block_num'] % 12

#number of days in the month 
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)
#dropping intial 12 months data
matrix = matrix[matrix.date_block_num > 11]
X_train = matrix[matrix['date_block_num'] < 34].drop('item_cnt_month', axis = 1)
y_train = matrix[matrix['date_block_num'] < 34]['item_cnt_month']

X_test = matrix[matrix['date_block_num'] == 34].drop('item_cnt_month', axis = 1)
y_test = matrix[matrix['date_block_num'] == 34]['item_cnt_month']
del matrix
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect();
%%time
model = XGBRegressor(max_depth=6, tree_method='gpu_hist', n_estimators=1000, colsample_bytree=0.8, subsample=0.8, eta=0.1,seed=42)

model.fit(X_train, y_train)
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
submission.head()
submission['item_cnt_month'] = model.predict(X_test)
submission.to_csv('Xgboost.txt', index = False)
