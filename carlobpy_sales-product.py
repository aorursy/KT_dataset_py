!ls ../input/*
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#dataviz

import matplotlib.pyplot as plt

import seaborn as sns

#manipulating date formats

import datetime

%matplotlib inline

# settings

import warnings

warnings.filterwarnings("ignore")
#import all datasets

salesdata = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
salesdata.head()
salesdata.info()
itemdf = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
itemdf.head()
itemcat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
iteminfo = pd.merge(itemdf, itemcat, on='item_category_id')
iteminfo.head()
salesitemdata = pd.merge(salesdata, iteminfo, on='item_id')
salesitemdata.head()
salesitemdata = salesitemdata.drop(['item_price','date','item_category_name','item_name'], axis=1)
salesitemdata.head()
salesitemdata['shop_id'] = salesitemdata['shop_id'].apply(int)

salesitemdata['item_id'] = salesitemdata['item_id'].apply(int)

salesitemdata['item_category_id'] = salesitemdata['item_category_id'].apply(int)
groupSalesDf = salesitemdata.groupby(['date_block_num','item_id','item_category_id','shop_id']).sum()

groupSalesDf['item_cnt_month']=groupSalesDf['item_cnt_day']

groupSalesDf.drop('item_cnt_day', axis=1, inplace=True)

groupSalesDf.reset_index(inplace=True)

groupSalesDf.drop('item_category_id', axis=1, inplace=True)

groupSalesDf
groupSalesDf.info()
# clip out sales of more than 20 in a month

clippedSales = groupSalesDf.copy()

clippedSales['item_cnt_month'].clip(0,20, inplace=True)

clippedSales['item_cnt_month'].max()
testdata = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv', index_col=0)
testdata
testdata['date_block_num'] = '34'

testdata
testdata['date_block_num'] = testdata['date_block_num'].apply(int)
testdata.info()
sns.lineplot(x='date_block_num', y='item_cnt_month', data=groupSalesDf[groupSalesDf['item_id']==22167])
# number of items per cat 

x=itemdf.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:10].reset_index()

x

# #plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
# Split Train Validation Data by Time

X_train = groupSalesDf[0:1126386].drop('item_cnt_month',axis=1)

y_train = groupSalesDf['item_cnt_month'][0:1126386]

X_valid = groupSalesDf[1126386:].drop('item_cnt_month',axis=1)

y_valid = groupSalesDf['item_cnt_month'][1126386:]
# Build Model

from xgboost import XGBRegressor

model = XGBRegressor(early_stopping_rounds=5,

                     eval_set=[(X_valid, y_valid)],

                     objective='reg:squarederror',

                     verbose=False)

model.fit(X_train,y_train)
pred = model.predict(X_valid)

# metrics

from sklearn import metrics

np.sqrt(metrics.mean_squared_error(y_valid, pred))
#Tune XGBR

model = XGBRegressor(objective='reg:squarederror',

                    n_estimators=100,

                    learning_rate=0.01,

                    colsample_bytree=1,

                    gamma=1,

                    subsample=0.8,

                    max_depth=3,

                    early_stopping_rounds=5,

                    eval_set=[(X_valid, y_valid)],

                    n_jobs=-1,

                    random_state=101)
model.fit(X_train,y_train)
pred = model.predict(X_valid)

# metrics

from sklearn import metrics

np.sqrt(metrics.mean_squared_error(y_valid, pred))
# Merge Training and Validation Sets

full_X = pd.concat([X_train,X_valid],axis=0)

full_X.sort_index(inplace=True)



full_y = pd.concat([y_train,y_valid],axis=0)

full_y.sort_index(inplace=True)
# Train on full data

model = XGBRegressor(objective='reg:squarederror',

                    n_estimators=100,

                    learning_rate=0.01,

                    colsample_bytree=1,

                    gamma=1,

                    subsample=0.8,

                    max_depth=3,

                    early_stopping_rounds=5,

                    n_jobs=-1,

                    random_state=101)

model.fit(full_X,full_y)
testdata = testdata[full_X.columns]
fullPred = model.predict(testdata)
presubmissionDf = pd.DataFrame(fullPred, columns=['item_cnt_month'])

presubmissionDf = pd.concat([testdata,presubmissionDf], axis=1)

presubmissionDf
submissionDf=pd.DataFrame(presubmissionDf['item_cnt_month'], columns=['item_cnt_month'])

submissionDf = submissionDf.rename_axis('ID')

submissionDf
submissionDf['item_cnt_month'].clip(0,20, inplace=True)

submissionDf['item_cnt_month'].max()
submissionDf.to_csv('Submission.csv')