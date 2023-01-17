# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn import *
import datetime
import nltk


train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
item_categories.head()
train['date'] = pd.to_datetime(train['date'],format = '%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date'],axis=1)

train = train.drop(['item_price'],axis=1)

train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']],as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns = {'item_cnt_monnth':'item_cnt_month'})
train.head()
shop_item_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'],as_index=False)['item_cnt_month'].mean()
shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
train = pd.merge(train,shop_item_mean,how='left',on=['shop_id','item_id'])
train.head()
shop_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})

shop_prev_month.head()
train = pd.merge(train,shop_prev_month,how='left',on=['shop_id','item_id']).fillna(0)
train = pd.merge(train,items,how='left',on=['item_id'])

train = pd.merge(train,item_categories,how='left',on=['item_category_id'])

train = pd.merge(train,shops,how='left',on=['shop_id'])
train.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test = pd.merge(test,shop_item_mean,how='left',on=['shop_id','item_id']).fillna(0.)
test = pd.merge(test,shop_prev_month,how='left',on=['shop_id','item_id']).fillna(0.)


test = pd.merge(test,items,how='left',on=['item_id'])

test = pd.merge(test,item_categories,how='left',on=['item_category_id'])

test = pd.merge(test,shops,how='left',on=['shop_id'])

test['item_cnt_month'] = 0.
test.head()
for c in ['item_name','shop_name','item_category_name']:
    
    lbl = sklearn.preprocessing.LabelEncoder()
    lbl.fit(list(train[c].unique())+list(test[c].unique()))
    train[c] = lbl.transform(train[c].astype(str))
    test[c] = lbl.transform(test[c].astype(str))
    print(c)
    
    
train.head()
# prepare date
col = [c for c in train.columns if c not in ['item_cnt_month']]

x1 =  train[train['date_block_num']<33]
y1 = np.log1p(x1['item_cnt_month'].clip(0.,20.))
x1 = x1[col]

x2 =  train[train['date_block_num']==33]
y2 = np.log1p(x2['item_cnt_month'].clip(0.,20.))
x2 = x2[col]
reg = ensemble.ExtraTreesRegressor(n_estimators=30,n_jobs=-1,max_depth=15,random_state=18)
reg.fit(x1,y1)

reg.fit(train[col],train['item_cnt_month'].clip(0.,20.))
test['item_cnt_month'] = reg.predict(test[col].clip(0.,20.))
test[['ID','item_cnt_month']].to_csv('submission.csv',index=False)
test['item_cnt_month'] = np.expm1(test['item_cnt_month'])
test[['ID','item_cnt_month']].to_csv('final_submission.csv',index=False)