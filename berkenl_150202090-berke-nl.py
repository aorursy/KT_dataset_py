import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from math import ceil

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt
sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

sales_test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

item_catalog = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv") 
sales_train.head(10)
sales_test.head(10)
items.head(10)
plt.figure(figsize=(10,4))

sns.scatterplot(x=sales_train.item_cnt_day, y=sales_train.item_price, data=sales_train)
items=items.drop("item_name",axis=1)
items.head(10)
sales_train = sales_train[sales_train.item_price<40000]

sales_train = sales_train[sales_train.item_cnt_day<300]

sales_train = sales_train[sales_train.item_cnt_day>=0]
plt.figure(figsize=(10,4))

sns.scatterplot(x=sales_train.item_cnt_day, y=sales_train.item_price, data=sales_train)
training = pd.merge(sales_train, items, on='item_id')

trained = training
training.head(10)
median = trained[(trained.shop_id==32)&(trained.item_id==2973)&(trained.date_block_num==4)&(trained.item_price>0)].item_price.median()

trained.loc[trained.item_price<0, 'item_price'] = median
training.date = training.date.apply(lambda x:datetime.datetime.strptime(x, "%d.%m.%Y"))

trained.head()
grouped = pd.DataFrame(training.groupby(['shop_id', 'date_block_num','item_id'])['item_cnt_day'].sum().reset_index())

total_item_cnt_mounth = grouped.groupby('date_block_num')['item_cnt_day'].sum()

grouped.head(10)
total_item_cnt_mounth_np = total_item_cnt_mounth.to_numpy()

total_item_cnt_mounth_np

mounths = np.arange(34)

X_train, X_test, y_train, y_test = train_test_split(mounths,total_item_cnt_mounth_np, test_size = 1/3, random_state = 12, shuffle=1)

X_train = X_train.reshape(-1, 1)

X_test = X_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)
model = LinearRegression(True,True,None,False)

model.fit(X_train, y_train)
model.predict([[34]])
pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, pred))

rmse
pred = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': pred.flatten()})

df
submission = df

submission.to_csv('submission.csv', index=True)

submission.head(50)
plt.figure(figsize=(10,4))

sns.scatterplot(x='Actual', y='Predicted', data=submission)
mean_squared_error(y_test, pred)