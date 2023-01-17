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

import numpy as np

import matplotlib.pyplot as plt

import xgboost as xgb



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
import pandas as pd

itemcat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
sales.loc[sales.item_cnt_day < 0.0, 'item_cnt_day'] = 0



sales_sum_shopitem = sales.groupby(['shop_id', 'item_id', 'item_price', 'date_block_num'], as_index=False).sum()



X = sales_sum_shopitem.iloc[:,1:-1]

y = sales_sum_shopitem["item_cnt_day"]



X_train = sales_sum_shopitem.drop(['item_cnt_day', 'item_price', 'date_block_num'], axis=1)

y_train = sales_sum_shopitem.item_cnt_day

X_test = test.drop('ID', axis=1)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

lr_result = lin_reg.predict(X_test)



result = (lr_result).clip(0, 20)



submission = pd.DataFrame({"ID" : test.index, "item_cnt_month" : result})

submission.to_csv('submission.csv', index = False)