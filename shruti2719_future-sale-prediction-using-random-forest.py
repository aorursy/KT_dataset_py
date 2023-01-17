import numpy as np

import pandas as pd

from pandas import Series, DataFrame

#data Visualization

import seaborn as sns

import matplotlib.pyplot as plt

import os



import sklearn

from sklearn import *

import nltk,datetime

from sklearn import ensemble, metrics, preprocessing
itemcat=pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items=pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

train=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

shops=pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

result=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
itemcat.info()
items.info()
train.info()
shops.info()
test.info()
result.info()
train.head()
train.date_block_num.max()
test.head()
train.describe()
print("train",train.shape)
print("test",test.shape)
train['date']= pd.to_datetime(train['date'], format='%d.%m.%Y')

train['month']=train['date'].dt.month

train['year']=train['date'].dt.year

train=train.drop(['date','item_price'],axis=1)

train=train.groupby([c for c in  train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()

train=train.rename(columns={'item_cnt_day':'item_cnt_month'})



train.head()
shop_item_mean= train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'],as_index=False)[['item_cnt_month']].mean()

shop_item_mean = shop_item_mean.rename(columns = {'item_cnt_month':'item_cnt_month_mean'})



#just add our mean feature to our train set



train = pd.merge(train, shop_item_mean, how='left',on=['shop_id', 'item_id'])



train.head()
#add last month

shop_prev_month= train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]

shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})

shop_prev_month.head()
#add previous month feature to train dataset

train = pd.merge(train, shop_prev_month, how= 'left', on =['shop_id','item_id']).fillna(0.)



#add all item features

train =  pd.merge(train, items, how= 'left', on='item_id')



#adding item category features

train = pd.merge(train,itemcat,how='left',on='item_category_id')



#adding shop faetures

train= pd.merge(train,shops,how='left', on='shop_id')



train.head()
#test dataset



#adding november 2015



test['month']=11

test['year']=2015

test['date_block_num']=34



#add mean feature

test = pd.merge(test,shop_item_mean,how='left',on=['shop_id','item_id']).fillna(0.)



#add previous month feature

test = pd.merge(test, shop_prev_month,how='left',on=['shop_id','item_id']).fillna(0.)



#add all the features

test= pd.merge(test,items,how='left',on=['item_id'])



#adding item category features

test=pd.merge(test,itemcat,how='left',on=['item_category_id'])



#adding shop features 

test =pd.merge(test,shops,how='left',on='shop_id')

test['item_cnt_month']=0

test.head()
#Label encoding

for c in ['shop_name','item_name','item_category_name']:

    lbl= preprocessing.LabelEncoder()

    lbl.fit(list(train[c].unique())+list(test[c].unique()))

    train[c]=lbl.transform(train[c].astype(str))

    test[c]=lbl.transform(test[c].astype(str))

    print(c)

#Lets train and predict using Random Forest algoritm



col = [c for c in train.columns if c not in ['item_cnt_month']]

x1=train[train['date_block_num']<33]

y1=np.log1p(x1['item_cnt_month'].clip(0.,20.)) #cliping values

x1=x1[col]

x2=train[train['date_block_num']==33]

y2=np.log1p(x2['item_cnt_month'].clip(0.,20.))

x2=x2[col]





reg=ensemble.ExtraTreesRegressor(n_estimators=30,n_jobs=-1,max_depth=20, random_state=18)

#no of trees are going to be in random forest=n_estimators

reg.fit(x1,y1)

print('RMSE value is:',np.sqrt(metrics.mean_squared_error(y2.clip(0.,20.),reg.predict(x2).clip(0.,20.))))
reg.fit(train[col],train['item_cnt_month'].clip(0.,20.))

test['item_cnt_month']=reg.predict(test[col]).clip(0.,20.)

test[['ID','item_cnt_month']].to_csv('result.csv',index=False)





test['item_cnt_month']=np.expm1(test['item_cnt_month'])

test[['ID','item_cnt_month']].to_csv('final_submission.csv',index=False)