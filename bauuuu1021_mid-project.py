import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



categ = pd.read_csv('../input/item_categories.csv')

item = pd.read_csv('../input/items.csv')

shop = pd.read_csv('../input/shops.csv')

train = pd.read_csv('../input/sales_train.csv')
train[['date', 'date_block_num']]
train[['item_price']].plot()

plt.show()



# remove extreme value

train = train[train.item_price < 250000]

train = train[train.item_price > 0]
train[['item_cnt_day']].plot()

plt.show()



# remove extreme value

train = train[train.item_cnt_day < 1500]

train = train[train.item_cnt_day > 0]

train
import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import train_test_split
test = pd.read_csv('../input/test.csv')
x = train[['date_block_num','shop_id','item_id']]

y = train['item_price']

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=42)
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_valid, y_valid)], 

    verbose=True, 

    early_stopping_rounds = 3)
test['date_block_num'] = 34

test_predict = test.drop(['ID'], axis=1)

test_predict = test_predict[['date_block_num','shop_id','item_id']]

predict_price = model.predict(test_predict)

test_predict['item_price'] = predict_price
'''

TODO : (next line) fix date_block_num, item_price 

'''

x = train.drop(['item_cnt_day','date'], axis=1) 

y = train['item_cnt_day']

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=42)

x_valid, x_drop, y_valid, y_drop = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_valid, y_valid)], 

    verbose=True, 

    early_stopping_rounds = 3)
result = model.predict(test_predict).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": result

})

submission.to_csv('submission.csv', index=False)