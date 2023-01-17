# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# /kaggle/input/competitive-data-science-predict-future-sales/shops.csv

# /kaggle/input/competitive-data-science-predict-future-sales/items.csv

# /kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv

# /kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sales_train['month'] = pd.to_datetime(sales_train['date']).dt.month

sales_train['year'] = pd.to_datetime(sales_train['date']).dt.year

sales_train.head()

# !pip3 --no-cache-dir install seaborn

import seaborn as sns

sns.set_style("darkgrid")
month_year_group = sales_train.groupby(['month','year']).agg({'item_cnt_day':'sum'}).reset_index()

sns.lineplot(x="month", y="item_cnt_day",style = "year",

             data=month_year_group,markers=True, dashes=False)
month_year_group = sales_train.groupby(['month','year']).agg({'item_price':'mean'}).reset_index()

sns.lineplot(x="month", y="item_price",style = "year",

             data=month_year_group,markers=True, dashes=False)
sales_train['total_sale'] = sales_train['item_cnt_day']*sales_train['item_price']

total_sale_group = sales_train.groupby(['month','year'])['total_sale'].sum().reset_index()

# total_sale_group.head()

sns.lineplot(x="month", y="total_sale",style="year",

             data=total_sale_group,markers=True, dashes=True)
#checking distribution for outliers in item_cnt_day and item_price

sns.boxplot(x=sales_train['item_price'])

sns.boxplot(x=sales_train['item_cnt_day'])
print("Shape before removing less then 0 or greater then 45000 item prices",sales_train.shape)

sales_train = sales_train[(sales_train.item_price > 0) & (sales_train.item_price < 45000)]

print("Shape after removing less then 0 or greater then 45000 item prices",sales_train.shape)
print("Shape before removing less then 0 or greater then 800 item_cnt_day",sales_train.shape)

sales_train = sales_train[(sales_train.item_cnt_day > 0) & (sales_train.item_cnt_day < 800)]

print("Shape after removing less then 0 or greater then 800 item_cnt_day",sales_train.shape)
#removing shops which are not in test set

print("Shape before removing shops and items which are not in test set",sales_train.shape)

sales_train = sales_train[sales_train.shop_id.isin(test_data.shop_id.unique())]

sales_train = sales_train[sales_train.item_id.isin(test_data.item_id.unique())]

print("Shape after removing shops and items which are not in test set",sales_train.shape)
sns.boxplot(x=sales_train['item_price'])
sns.boxplot(x=sales_train['item_cnt_day'])
month_year_group = sales_train.groupby(['month','year']).agg({'item_cnt_day':'sum'}).reset_index()

sns.lineplot(x="month", y="item_cnt_day",style = "year",

             data=month_year_group,markers=True, dashes=False)
month_year_group = sales_train.groupby(['month','year']).agg({'item_price':'mean'}).reset_index()

sns.lineplot(x="month", y="item_price",style = "year",

             data=month_year_group,markers=True, dashes=False)
sales_train['total_sale'] = sales_train['item_cnt_day']*sales_train['item_price']

total_sale_group = sales_train.groupby(['month','year'])['total_sale'].sum().reset_index()

# total_sale_group.head()

sns.lineplot(x="month", y="total_sale",style="year",

             data=total_sale_group,markers=True, dashes=True)
sales_train_monthly = sales_train.groupby(['date_block_num','item_id','shop_id']).agg({'item_cnt_day':'sum'}).reset_index()
pivoted_train_data = sales_train_monthly.pivot(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day').fillna(0).reset_index()
pivoted_train_data.shape
test_dataset = pd.merge(pivoted_train_data,test_data,left_on = ['shop_id','item_id'],right_on = ['shop_id','item_id'],how = 'right').fillna(0)
test_dataset.shape
test_data.shape
test_dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)
#why are we expanding it ?

X_train = np.expand_dims(test_dataset.values[:,:-1],axis = 2)

y_train = test_dataset.values[:,-1:]



X_test = np.expand_dims(test_dataset.values[:,1:],axis = 2)



 

print(X_train.shape,y_train.shape,X_test.shape)
# print(X_train.shape,y_train.shape,X_test.shape)
from keras.models import Sequential

from keras.layers import Dense,Dropout,LSTM



model = Sequential()

model.add(LSTM(64,input_shape=(X_train.shape[1],X_train.shape[2])))

model.add(Dropout(0.5))

model.add(Dense(1))



model.summary()
model.compile(loss='mse',optimizer='adam',metrics = ['mean_squared_error'])
model.fit(X_train,y_train,batch_size = 32,epochs = 10)
submit_data = model.predict(X_test)

submission = pd.DataFrame({'id':test_data['ID'],'item_cnt_month':submit_data.ravel()})

submission.clip(0,20)

submission.head()
submission.to_csv('test_sub.csv',index = False)