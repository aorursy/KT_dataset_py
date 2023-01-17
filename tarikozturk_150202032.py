# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics 

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales_data = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
sales_data.head()
itemcategories_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
itemcategories_data.info()

itemcategories_data.head()
items_data.info()

items_data.head()
shops_data.info()

shops_data.head()
test_data.info()

test_data.head()
shops_data.isnull().sum()

sales_data.isnull().sum()

plt.figure(figsize=(10,4))

sns.scatterplot(x=sales_data.item_cnt_day, y=sales_data.item_price, data=sales_data)
sales_data = sales_data[sales_data.item_price<45000]

sales_data = sales_data[sales_data.item_cnt_day<600]
plt.figure(figsize=(10,4))

sns.scatterplot(x=sales_data.item_cnt_day, y=sales_data.item_price, data=sales_data)
sales_train_sub = sales_data

sales_train_sub['month'] = pd.DatetimeIndex(sales_train_sub['date']).month

sales_train_sub['year'] = pd.DatetimeIndex(sales_train_sub['date']).year

sales_train_sub.head(10)
satıs_grup = sales_train_sub.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()



x=satıs_grup.iloc[:,:-1]

y=satıs_grup.iloc[:,-1:]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)
x
from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor(n_estimators=25,random_state=16)

etr.fit(x_train,y_train.values.ravel())

y_pred = etr.predict(x_test)





print("R2 Score:",r2_score(y_test,y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))