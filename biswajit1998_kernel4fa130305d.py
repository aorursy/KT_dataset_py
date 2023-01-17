# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/competitive-data-science-predict-future-sales/'
sales_train = pd.read_csv(path+'sales_train.csv')

sales_train.head(5)
item_categories = pd.read_csv(path+'item_categories.csv')

item_categories.head()
shops = pd.read_csv(path+'shops.csv')

shops.head()
items = pd.read_csv(path+'items.csv')

items.head()
test = pd.read_csv(path+'test.csv')

test.head()
test['date_block_num'] = 34
test = test.merge(items,how='left',on='item_id')
test.drop(['item_name'], axis=1,inplace= True)
test.head()
sales_train = sales_train.merge(shops,how='left',on='shop_id')
sales_train = sales_train.merge(items,how='left',on='item_id')
sales_train = sales_train.merge(item_categories,how='left',on='item_category_id')
sales_train.head()
sales_train['total_price_per_day'] = sales_train.item_cnt_day * sales_train.item_price
sales_train.head()
sales_train[['day','month','year']] = sales_train.date.str.split(".",expand=True)
sales_train.head()
sales_train_shop_item = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].sum()
sales_train_shop_item = sales_train_shop_item.reset_index()
sales_train_shop_item.rename(columns={"item_cnt_day":"item_cnt_month"}, inplace=True)

sales_train_shop_item.rename(columns={"total_price_per_day":"total_price_per_month"}, inplace=True)
sales_train_shop_item.head()
sales_min = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].min()
sales_min = sales_min.reset_index()
sales_min.head()
sales_train_shop_item['min_item_cnt_month'] = sales_min['item_cnt_day']

sales_train_shop_item['min_price_per_month'] = sales_min['total_price_per_day']
sales_train_shop_item.head()
sales_max = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].max()
sales_max = sales_max.reset_index()
sales_train_shop_item['max_item_cnt_month'] = sales_max['item_cnt_day']

sales_train_shop_item['max_price_per_month'] = sales_max['total_price_per_day']
sales_train_shop_item.head()
sales_avg = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].mean()
sales_avg = sales_avg.reset_index()
sales_train_shop_item['avg_item_cnt_month'] = sales_avg['item_cnt_day']

sales_train_shop_item['avg_price_per_month'] = sales_avg['total_price_per_day']
test.head()
sales_train_shop_item.head()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id"]]

y = sales_train_shop_item["total_price_per_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id']])

y_test = reg.predict(pred)

print(y_test)
test['total_price_per_month'] = y_test
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month"]]

y = sales_train_shop_item["min_item_cnt_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month']])

y_test = reg.predict(pred)

print(y_test)
test['min_item_cnt_month'] = y_test
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","min_item_cnt_month"]]

y = sales_train_shop_item["min_price_per_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month',"min_item_cnt_month"]])

y_test = reg.predict(pred)

print(y_test)
test['min_price_per_month'] = y_test
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","min_price_per_month"]]

y = sales_train_shop_item["max_item_cnt_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month','min_price_per_month']])

y_test = reg.predict(pred)

print(y_test)
test['max_item_cnt_month'] = y_test
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","max_item_cnt_month"]]

y = sales_train_shop_item["max_price_per_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month',"max_item_cnt_month"]])

y_test = reg.predict(pred)

print(y_test)
test['max_price_per_month'] = y_test
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","max_item_cnt_month","min_item_cnt_month"]]

y = sales_train_shop_item["avg_item_cnt_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month','max_item_cnt_month','min_item_cnt_month']])

y_test = reg.predict(pred)

print(y_test)
test['avg_item_cnt_month'] = y_test
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","avg_item_cnt_month"]]

y = sales_train_shop_item["avg_price_per_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month',"avg_item_cnt_month"]])

y_test = reg.predict(pred)

print(y_test)
test['avg_price_per_month'] = y_test
test.head()
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","min_item_cnt_month","min_price_per_month","max_item_cnt_month","max_price_per_month","avg_item_cnt_month","avg_price_per_month"]]

y = sales_train_shop_item["item_cnt_month"]

poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X)

reg = LinearRegression().fit(X_, y)

pred = poly.fit_transform(test[["date_block_num", "shop_id", "item_id","total_price_per_month","min_item_cnt_month","min_price_per_month","max_item_cnt_month","max_price_per_month","avg_item_cnt_month","avg_price_per_month"]])

y_test = reg.predict(pred)

print(y_test)
submit = pd.DataFrame({'ID':np.arange(len(y_test)),'item_cnt_month':np.clip(y_test, a_min = 0, a_max = 20)},columns=['ID','item_cnt_month'])
submit.head()
submit.to_csv('submission.csv',index = False)
# sample_submission = pd.read_csv(path+'sample_submission.csv')

# sample_submission.head()
# from sklearn.linear_model import LinearRegression

# from sklearn.preprocessing import PolynomialFeatures

# from xgboost import XGBRegressor
# sales_train_shop_item.head()
# X = sales_train_shop_item[["date_block_num", "shop_id", "item_id"]]

# y = sales_train_shop_item["item_cnt_month"]

# # poly = PolynomialFeatures(degree=3)

# # X_ = poly.fit_transform(X)

# xgb_model = XGBRegressor(max_depth=8, 

#                          n_estimators=500, 

#                          min_child_weight=1000,  

#                          colsample_bytree=0.7, 

#                          subsample=0.7, 

#                          eta=0.3, 

#                          seed=0)

# xgb_model.fit(X, y, eval_metric="rmse")
# reg = LinearRegression().fit(X_, y)
# test.head(5)
# y_test = xgb_model.predict(test[['date_block_num','shop_id','item_id']])

# # pred = poly.fit_transform(test[['date_block_num','shop_id','item_id']])
# y_test = reg.predict(pred)
# y_test
# submit = pd.DataFrame({'ID':np.arange(len(y_test)),'item_cnt_month':np.clip(y_test, a_min = 0, a_max = 20)},columns=['ID','item_cnt_month'])
# submit.describe()
# submit.to_csv('submission.csv',index = False)