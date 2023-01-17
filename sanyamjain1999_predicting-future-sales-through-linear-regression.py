import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

s_train = pd.read_csv("../input/dataset/sales_train_v2.csv")

s_sub = pd.read_csv("../input/dataset/sample_submission.csv")

test = pd.read_csv("../input/dataset/test.csv")
item_categories.head(5)
item_categories.tail(5)
items.head(5)
items.tail(5)
shops.head(5)
shops.tail(5)
s_train.head(5)
s_train.tail(5)
s_sub.head(5)
test.head(5)
test.tail(5)
s_train = s_train.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
s_train.head(5)
from datetime import datetime, date

from dateutil.relativedelta import relativedelta



s_train['month'] = s_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))

s_train['year'] = s_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))
s_train.head(5)
s_train = s_train.drop('date', axis=1)

s_train = s_train.drop('item_category_id', axis=1)
s_train.head(5)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
s_train.head(5)
s_train.groupby(['item_id', 'date_block_num', 'shop_id']).mean()
s_train
X_train,Y_train = s_train.drop(["month", "year", "item_price", "item_cnt_day"],axis=1),s_train.item_cnt_day
X_train
Y_train
X_train = X_train.values

X_train
Y_train = Y_train.values

Y_train
pred = test
test = test.values

test
model = LinearRegression()

model.fit(X_train, Y_train)
y_test = model.predict(test)
y_test = y_test.astype(int)
pred = pred.drop(['shop_id', 'item_id'], axis=1)
pred['item_cnt_month'] = y_test
pred.to_csv('submission.csv', index=False)