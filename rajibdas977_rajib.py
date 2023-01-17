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
sales_train.describe()
item_categories = pd.read_csv(path+'item_categories.csv')

item_categories.head()
shops = pd.read_csv(path+'shops.csv')

shops.head()
items = pd.read_csv(path+'items.csv')

items.head()
test = pd.read_csv(path+'test.csv')

test.head()
sales_train['total_price'] = sales_train.item_cnt_day * sales_train.item_price
sales_train_shop_item = sales_train.groupby(by=['date_block_num','shop_id','item_id'])[['item_cnt_day','total_price']].sum()
sales_train_shop_item = sales_train_shop_item.reset_index()
sales_train_shop_item.rename(columns={"item_cnt_day":"item_cnt_month"}, inplace=True)
sales_train.head()
sales_train_shop_item.describe()
# from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import PolynomialFeatures
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id"]]

# # y = sales_train_shop_item["item_cnt_month"]

y = sales_train_shop_item.total_price

reg = ElasticNet(random_state=0).fit(X, y)

# poly = PolynomialFeatures(degree=3)

# X_ = poly.fit_transform(X)
# reg = LinearRegression().fit(X_, y)
test['date_block_num'] = 34
test.head(5)
# pred = poly.fit_transform(test[['date_block_num','shop_id','item_id']])
y_test = reg.predict(test[['date_block_num','shop_id','item_id']])
y_test
test['total_price'] = y_test
test.describe()
X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price"]]

y = sales_train_shop_item["item_cnt_month"]

poly = PolynomialFeatures(degree=2)

X_ = poly.fit_transform(X)
# reg_1 = LinearRegression().fit(X_, y)

reg_1 = ElasticNet(random_state=0).fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price']])
y_test = reg_1.predict(pred)
y_test
submit = pd.DataFrame({'ID':np.arange(len(y_test)),'item_cnt_month':np.clip(y_test, a_min = 0, a_max = 20)},columns=['ID','item_cnt_month'])
# submit.item_cnt_month = submit["item_cnt_month"].astype(int)
submit.describe()
submit.to_csv('submission.csv',index = False)