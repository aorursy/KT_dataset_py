# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
train.head()
ts=train.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts);
plt.figure(figsize=(20,5))



index = [str(i) for i in train["shop_id"].value_counts().index]

value = train["shop_id"].value_counts()

plt.bar(index, value, color="blue", alpha=0.5)

plt.xlabel("Shop_ID")

plt.ylabel("Item_Counts")

plt.title("ShopID/ItemCounts Visulation")

train = train[train.item_price <= 100000]

train = train[train.item_cnt_day <= 1000]

train
test_shops = test.shop_id.unique()

train = train[train.shop_id.isin(test_shops)]

test_items = test.item_id.unique()

train = train[train.item_id.isin(test_items)]

train.head()
subDroppedDf = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']

train.drop_duplicates(subDroppedDf,keep='first', inplace=True) 

train.reset_index(drop=True, inplace=True)
ax = sns.distplot(train.groupby('date_block_num').sum()['item_cnt_day'], color="navy")

plt.show()
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
dataset = train.pivot_table(index=['item_id', 'shop_id'], values=['item_cnt_day'],columns='date_block_num', fill_value=0)

dataset = dataset.reset_index()

dataset.head()
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')

dataset = dataset.fillna(0)

dataset.head()
dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)

dataset.head()
y_t = dataset.iloc[:,-1:]



x_t = dataset.iloc[:,1:]

x_t.drop(dataset.iloc[:,-1:], axis = 1, inplace = True)



x_t = (x_t - np.min(x_t)) / (np.max(x_t) - np.min(x_t))
x_t
y_t
x = x_t.iloc[:, :]

y = y_t.values.reshape(-1,1)

# y.astype(int)
x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = LinearRegression()  

model.fit(x, y) 



print(model.score(x_train, y_train))
print(model.intercept_)

print(model.coef_)
pred = model.predict(x_test)
mae = metrics.mean_absolute_error(model.predict(x_test), y_test)

mse = metrics.mean_squared_error(model.predict(x_test), y_test)

rmse = np.sqrt(mse)



print('Mean Absolute Error (MAE): %.2f' % mae)

print('Mean Squared Error (MSE): %.2f' % mse)

print('Root Mean Squared Error (RMSE): %.2f' % rmse)
plt.figure(figsize=(15,8))

plt.scatter(y_test, model.predict(x_test), color='r')

plt.xlabel("Actual Values:")

plt.ylabel("Predicted Valuess:")

plt.title("Actual vs Predicted Values:")

plt.show()