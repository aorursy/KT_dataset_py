# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import os 

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        





        
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates = ['date'])

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

data_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')



sales_train.info()

sales_train.head()
plt.figure(figsize=(10,10))

plt.scatter(sales_train.item_cnt_day,sales_train.item_price)

plt.show()
sales_train = sales_train[sales_train.item_price<40000]

sales_train = sales_train[sales_train.item_cnt_day<200]

columns = ['date', 'date_block_num', 'shop_id', 'item_id','item_price','item_cnt_day']

sales_train.drop_duplicates(columns,keep='first', inplace=True)



plt.figure(figsize=(10,10))

plt.scatter(sales_train.item_cnt_day,sales_train.item_price)

plt.show()


data = sales_train[['item_cnt_day','item_price']]

data.info()

data.head()
x = data.iloc[:, :-1].values

y = data.iloc[:, 1].values



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 123, shuffle=1)



model = LinearRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, model.predict(X_train), color = 'blue')

plt.xlabel('item_cnt_day')

plt.ylabel('item_price')

plt.show()





r2_score(y_test, y_pred)