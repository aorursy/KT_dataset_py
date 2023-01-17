#Author: dharmendra tolani

#Drop item_id from both train/test - replace with Leave one out feature which is total - current item count
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from sklearn import linear_model

import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import xgboost as xgb

import time

import datetime
items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')

cats = pd.read_csv('../input/item_categories.csv')

sales = pd.read_csv('../input/sales_train.csv')

test  = pd.read_csv('../input/test.csv')
X = sales.copy()



#fixing outliers - https://www.kaggle.com/dlarionov/feature-engineering-xgboost

X = X[X.item_price<100000]

X = X[X.item_cnt_day<1001]

median = X[(X.shop_id==32)&(X.item_id==2973)&(X.date_block_num==4)&(X.item_price>0)].item_price.median()

X.loc[X.item_price<0, 'item_price'] = median



#how many items were sold for each shop_id,item_id,month combination

g2 = X.groupby(['shop_id','item_id','date_block_num'], as_index=False).agg({'item_cnt_day': lambda x: x.sum()}).sort_values(['date_block_num'], ascending=True)

g2.head()
ts = time.time()

rows_list = []

shop_item = {}

shop_item_month = {}



for index,row in g2.iterrows():

    shop_id = row['shop_id']

    item_id = row['item_id']

    date_block_num = row['date_block_num']

    item_cnt = row['item_cnt_day']        

    shop_item[str(shop_id)+"_"+str(item_id)] = 1

    shop_item_month[str(shop_id)+"_"+str(item_id)+"_"+str(date_block_num)] = 1

    dict = {}

    dict["shop_id"] = shop_id

    dict["item_id"] = item_id

    dict["date_block_num"] = date_block_num

    dict["item_cnt"] = item_cnt

    rows_list.append(dict)

print(time.time() - ts)

str(datetime.datetime.now())
#for every shop_id,item_id combination present in the train data - add 0 for missing months - resulted in result improvement

for shop_item_key in shop_item:

    result = shop_item_key.split("_")

    shop_id = result[0]

    item_id = result[1]

    for month in range(34):

        shop_item_month_key = shop_item_key+"_"+str(month)

        if(shop_item_month_key not in shop_item_month):

            dict = {}

            dict["shop_id"] = shop_id

            dict["item_id"] = item_id

            dict["date_block_num"] = month

            dict["item_cnt"] = 0

            rows_list.append(dict)



#monthly training data with empty months handled(_eh)

monthly_train_data_eh_original = pd.DataFrame(rows_list) #snapshot

monthly_train_data_eh = monthly_train_data_eh_original.copy()

monthly_train_data_eh.head()
CLIP_TO = 200

#clipping the target variable (item sales count) to 200. Tried with multiple values.

#improved score, 200 > 100 > 60 > 40 > 30 > 26 > 24 > 22 > 20 >  18 (ranking)

# increasing the above threshold beyond 200 backfires, tried 400,300,250,220,205
#leave one out - drop item_id after aggregating item_id - best so far

#logic

#aggregate sales for item_id

#add new column 'loo'

#for training data, subtract sales of current row from total

#for test data, keep entire sales

#drop item_id from both test and train dataset
ts = time.time()

monthly_train_data_eh['item_cnt'] = monthly_train_data_eh['item_cnt'].clip(0,CLIP_TO)

monthly_train_data_eh = monthly_train_data_eh.sort_values(['shop_id', 'item_id', 'date_block_num'], ascending=[True, True, True])



#prev 2 month sales as features (lag features)

monthly_train_data_eh['prev_month_sales'] = monthly_train_data_eh['item_cnt'].shift()

monthly_train_data_eh['prev_month_sales_minus_1'] = monthly_train_data_eh['prev_month_sales'].shift()



monthly_train_data_eh['c'] = monthly_train_data_eh['item_cnt']



#aggregating sales by item_id

g2 = monthly_train_data_eh.groupby(['item_id'], as_index=False).agg({'c': lambda x: x.sum()})

g2 = g2.rename(columns={'c': 'total_sales'})



monthly_train_data_eh = monthly_train_data_eh.merge(g2, how='left')

monthly_train_data_eh.head()
#loo is short for leave one out - idea is to add a feature which represents total_item_sales - current_row_sales

#read more about leave one out online.

monthly_train_data_eh['loo_train'] = monthly_train_data_eh['total_sales'] - monthly_train_data_eh['c']
ts = time.time()

month_33 = monthly_train_data_eh.loc[(monthly_train_data_eh['date_block_num'] == 33)].copy()

month_33['item_id'] = month_33['item_id'].astype(float)

month_33['item_id'] = month_33['item_id'].astype(int)

month_33.drop_duplicates(keep='first', subset='item_id', inplace=True)

month_33.head()
monthly_train_data_eh.drop(['total_sales'],axis=1, inplace=True) #now this column is no longer required

month_33.head()
month_33_bkp = month_33.copy() #snapshot
monthly_train_data_eh['item_id'] = monthly_train_data_eh['item_id'].astype(float)

monthly_train_data_eh['item_id'] = monthly_train_data_eh['item_id'].astype(int)

monthly_train_data_eh['shop_id'] = monthly_train_data_eh['shop_id'].astype(float)

monthly_train_data_eh['shop_id'] = monthly_train_data_eh['shop_id'].astype(int)
#test data preparation

x_test_df = test.copy()

x_test_df = pd.concat([x_test_df.set_index('item_id'),items.set_index('item_id')], axis=1, join='inner').reset_index()

x_test_df.head()
x_test_df = x_test_df.drop(['item_name'], axis=1)

x_test_df['date_block_num'] = 34

cols = ['date_block_num', 'item_id', 'shop_id']

x_test_df = x_test_df[cols]

x_test_df['shop_id'] = (x_test_df['shop_id']).astype(int)

x_test_df['item_id'] = (x_test_df['item_id']).astype(int)

x_test_df.head()
#temporarily set month to 33 so that we can merge with corresponding train data

x_test_df['date_block_num'] = 33

month_33['item_id'] = month_33['item_id'].astype(int)

month_33['date_block_num'] = month_33['date_block_num'].astype(int)

month_33.head()
x_test_df_bkp = x_test_df.copy() #save a snapshot
x_test_df = x_test_df.merge(month_33, how='left', on=['item_id', 'date_block_num'])

x_test_df.head()
x_test_df['date_block_num'] = 34

#for test data total sales = leave one out, since there is no current sales to subtract

x_test_df = x_test_df.rename(columns={'total_sales': 'loo'})
#rename, drop columns

x_test_df.drop(['prev_month_sales_minus_1'],axis=1, inplace=True)

x_test_df = x_test_df.rename(columns={'c': 'prev_month_sales', 'prev_month_sales': 'prev_month_sales_minus_1'})



x_test_df = x_test_df.rename(columns={'shop_id_x': 'shop_id'})

x_test_df.drop(['shop_id_y'],axis=1, inplace=True)



monthly_train_data_eh = monthly_train_data_eh.rename(columns={'loo_train': 'loo'})

monthly_train_data_eh.drop(['c'],axis=1, inplace=True)



#fillna(0) led to result improvement

monthly_train_data_eh = monthly_train_data_eh.fillna(0)

x_test_df = x_test_df.fillna(0)



#drop item_id from both train and test datasets, as it's been replaced by new feature 'loo' hence no longer required

monthly_train_data_eh.drop(['item_id'],axis=1, inplace=True)

x_test_df.drop(['item_id'],axis=1, inplace=True)



#split train,valid data

monthly_valid_data_eh = monthly_train_data_eh[monthly_train_data_eh['date_block_num'] == 33].copy()

monthly_train_data_eh = monthly_train_data_eh[monthly_train_data_eh['date_block_num'] < 33]
y_cols = ['item_cnt']



y_train = monthly_train_data_eh[y_cols]

x_train = monthly_train_data_eh.copy()

x_train = x_train.drop(y_cols, axis=1)



y_valid = monthly_valid_data_eh[y_cols]

x_valid = monthly_valid_data_eh.copy()

x_valid = x_valid.drop(y_cols, axis=1)



x_train_df = pd.DataFrame(x_train)

y_train_df = pd.DataFrame(y_train)

x_valid_df = pd.DataFrame(x_valid)

y_valid_df = pd.DataFrame(y_valid)
x_test_df = x_test_df[list(x_train_df)] #ensure that train and test datasets have same columns
x_test_df['shop_id'] = (x_test_df['shop_id']).astype(int)

x_train_df['shop_id'] = (x_train_df['shop_id']).astype(int)

x_valid_df['shop_id'] = (x_valid_df['shop_id']).astype(int)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)



xg_reg.fit(

    x_train_df, 

    y_train_df, 

    eval_metric="rmse", 

    eval_set=[(x_train_df, y_train_df), (x_valid_df, y_valid_df)], 

    verbose=True, 

    early_stopping_rounds = 10)



preds = xg_reg.predict(x_test_df).clip(0,200)



test = test.set_index('ID')

submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": preds

})



submission.to_csv("submission_from_kernel.csv", mode = 'w', index=False)

print("file written")