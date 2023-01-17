import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from itertools import product

import datetime as dt

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
train_users = pd.read_csv('../input/train-dataset/sales_train_v2.csv')

test_users = pd.read_csv('../input/test-dataset/test.csv')

print("There were", train_users.shape[0], "observations in the training set and", test_users.shape[0], "in the test set.")

print("In total there were", train_users.shape[0] + test_users.shape[0], "observations.")
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()

shops['shop_city'], shops['shop_name'] = shops['shop_name'].str.split(' ', 1).str

shops.head()
plt.figure(figsize=(12,10))

cor = train_users.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_users.head(30)
test_users.head(10)
train_users.isnull().sum()
train_users['date'] = pd.to_datetime(train_users['date'], format='%d.%m.%Y')

train_users['Year'] = train_users['date'].dt.year

train_users['Month'] = train_users['date'].dt.month

train_users['Day'] = train_users['date'].dt.day

train_users.head(10)

train_users.dtypes
train_users.groupby(['shop_id'])['item_cnt_day'].aggregate('count').reset_index().sort_values('item_cnt_day', ascending=False)
def unique_counts(train_users):

   for i in train_users.columns:

       count = train_users[i].nunique()

       print(i, ": ", count)

unique_counts(train_users)
train_users.describe()
plt.figure(figsize=(12,6))

sns.countplot(x='Year', data = train_users)

plt.xlabel('Year')

plt.ylabel('item_cnt_day')

plt.title('Yearly item_cnt_day')

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='Month', data = train_users)

plt.xlabel('Month')

plt.ylabel('item_cnt_day')

plt.title('Monthly item_cnt_day')

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='Year', data = train_users, hue='Month')

plt.xlabel('Year')

plt.ylabel('item_cnt_day')

plt.title('Yearly item_cnt_day')

sns.despine()
plt.figure(figsize=(10,4))

plt.xlim(train_users.item_cnt_day.min(), train_users.item_cnt_day.max()*1.1)

sns.boxplot(x=train_users.item_cnt_day)
train_users=train_users[train_users['item_cnt_day']>0]

train_users=train_users[train_users['item_cnt_day']<700]
plt.figure(figsize=(10,4))

plt.xlim(train_users.item_price.min(), train_users.item_price.max()*1.1)

sns.boxplot(x=train_users.item_price)
train_users=train_users[train_users['item_price']>0]

train_users=train_users[train_users['item_price']<65000]
train_users = train_users.drop_duplicates(keep = 'first')

print('Number of duplicates:', len(train_users[train_users.duplicated()]))
p_df = train_users.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

p_df.head()
train_cleaned_df = p_df.reset_index()
train_cleaned_df
dataset = pd.merge(test_users,train_cleaned_df,on = ['item_id','shop_id'],how = 'left')
dataset.fillna(0,inplace = True)

dataset.head()
dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

dataset.head()
X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

y_train = dataset.values[:,-1:]

X_test = np.expand_dims(dataset.values[:,1:],axis = 2) 

print(X_train.shape,y_train.shape,X_test.shape)
my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1)))

my_model.add(Dropout(0.4))

my_model.add(Dense(1))

my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

my_model.summary()
my_model.fit(X_train,y_train,batch_size = 4096,epochs = 10)
submission_pfs = my_model.predict(X_test)

submission_pfs = submission_pfs.clip(0,20)

submission = pd.DataFrame({'ID':test_users['ID'],'item_cnt_month':submission_pfs.ravel()})

submission.to_csv('Submission.csv',index = False)