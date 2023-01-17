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
import pandas as pd

sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sales_train
def conv(x):

    if x < 0 :

        return(x*(-1))

    else:

        return(x)

sales_train['item_cnt_day'] = sales_train['item_cnt_day'].apply(conv)
sales_train = sales_train.drop(['date', 'item_price'], axis=1)
sales_train
test
train = sales_train.groupby(['date_block_num','shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
test['date_block_num'] = 34

test_set = test
test = test_set[['date_block_num', 'shop_id', 'item_id']]
test
train
train[train.shop_id.isin([59]) & train.item_id.isin([22102])]
X_train = train[train.date_block_num < 33].drop(['item_cnt_day'], axis=1)

Y_train = train[train.date_block_num < 33]['item_cnt_day']

X_valid = train[train.date_block_num == 33].drop(['item_cnt_day'], axis=1)

Y_valid = train[train.date_block_num == 33]['item_cnt_day']

X_test = test
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_valid)
Y_pred[10]
Y_valid.iloc[10]
from sklearn.metrics import mean_squared_error

from math import sqrt

rmse = sqrt(mean_squared_error(Y_pred, Y_valid))

rmse
X_train = train.drop(['item_cnt_day'], axis=1)

Y_train = train['item_cnt_day']

model.fit(X_train, Y_train)

Y_test = model.predict(X_test)
submission = pd.DataFrame({

    "ID": test_set.index, 

    "item_cnt_month": Y_test

})
submission