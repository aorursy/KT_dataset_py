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
import warnings

import matplotlib.pyplot as plt

from keras import optimizers

from keras.utils import plot_model

from keras.models import Sequential, Model

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten

from sklearn import metrics

from sklearn.model_selection import train_test_split

import math

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
train.head()
test.head()
train_agg = train.drop(['date', 'item_price'], axis=1)

train_agg.describe()
df = train_agg.groupby(["shop_id", "item_id", "date_block_num"])



monthly = df.aggregate({"item_cnt_day":np.sum}).fillna(0)

monthly.reset_index(level=["shop_id", "item_id", "date_block_num"], inplace=True)

monthly = monthly.rename(columns={ monthly.columns[3]: "item_cnt_month" })
monthly.describe()
monthly['item_id'].value_counts()/34
test['item_id'].value_counts()
monthly['shop_id'].loc[monthly['item_id'] == 5822].value_counts().sort_index()
test['shop_id'].loc[test['item_id'] == 5822].value_counts().sort_index()
monthly['shop_id'].value_counts()
test['shop_id'].value_counts()
monthly.describe()
test.describe()
train_simple = monthly.drop('date_block_num', axis=1)

#shuffle rows

train_simple = train_simple.sample(frac=1).reset_index(drop=True)



X_simple = train_simple[['shop_id', 'item_id']]

y_simple = train_simple['item_cnt_month']
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 214200

n_trn = len(train_simple) - n_valid

X_train, X_valid = split_vals(X_simple, n_trn)

y_train, y_valid = split_vals(y_simple, n_trn)
plt.scatter(X_valid.iloc[:100,1], y_valid[:100], color='black')
m = RandomForestRegressor(n_estimators=1, n_jobs=-1)

%time m.fit(X_train, y_train)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)



print_score(m)
plt.scatter(X_valid.iloc[:100,1], m.predict(X_valid)[:100], color='black')
m_2 = RandomForestRegressor(n_estimators=100, n_jobs=-1)

%time m_2.fit(X_train, y_train)

%time print_score(m_2)
preds = np.stack([t.predict(X_valid) for t in m_2.estimators_])
plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
X_simple = train_simple[['shop_id', 'item_id']]

y_simple = train_simple['item_cnt_month'].clip(0,20)
n_valid = 214200

n_trn = len(train_simple) - n_valid

X_train, X_valid = split_vals(X_simple, n_trn)

y_train, y_valid = split_vals(y_simple, n_trn)
m = RandomForestRegressor(n_estimators=1, n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m_2 = RandomForestRegressor(n_estimators=100, n_jobs=-1)

%time m_2.fit(X_train, y_train)

print_score(m_2)
preds = np.stack([t.predict(X_valid) for t in m_2.estimators_])

plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
pd.DataFrame(m_2.predict(X_valid)).describe()
train_td = monthly.sort_values(by=["date_block_num"])

valid_td = monthly[monthly["date_block_num"] == 33]



X_train = train_td[['shop_id', 'item_id']]

y_train = train_td['item_cnt_month'].clip(0,20)

X_valid = valid_td[['shop_id', 'item_id']]

y_valid = valid_td['item_cnt_month'].clip(0,20)
m_3 = RandomForestRegressor(n_estimators=60, n_jobs=-1)

%time m_3.fit(X_train, y_train)

print_score(m_3)
preds = np.stack([t.predict(X_valid) for t in m_3.estimators_])

plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(60)]);
import xgboost as xgb

param = {'max_depth':12,  # originally 10

         'subsample':1,  # 1

         'min_child_weight':0.5,  # 0.5

         'eta':0.3,

         'num_round':1000, 

         'seed':0,  # 1

         'silent':0,

         'eval_metric':'rmse',

         'early_stopping_rounds':100

        }



progress = dict()

xgbtrain = xgb.DMatrix(X_train, y_train)

watchlist  = [(xgbtrain,'train-rmse')]

m_4 = xgb.train(param, xgbtrain)


preds = m_4.predict(xgb.DMatrix(X_valid))



rmse = np.sqrt(mean_squared_error(preds, y_valid))

print(rmse)
new_submission = pd.merge(monthly, test, how='right', left_on=['shop_id','item_id'], right_on = ['shop_id','item_id']).fillna(0)

new_submission.drop(['shop_id', 'item_id'], axis=1)

new_submission = new_submission[['ID','item_cnt_month']]
new_submission['item_cnt_month'] = new_submission['item_cnt_month'].clip(0,20)

new_submission.describe()
new_submission.to_csv('previous_value.csv', index=False)
from keras.models import Sequential

from keras.layers import Dense, LSTM

from keras.optimizers import Adam

import tensorflow as tf
from datetime import datetime

train['year']=pd.to_datetime(train['date']).dt.strftime('%Y')

train['month']=pd.to_datetime(train['date']).dt.strftime('%m')

train['day']=pd.to_datetime(train['date']).dt.strftime('%d')

train.head()
import seaborn as sns
grouped = pd.DataFrame(train.groupby(['year','month'])['item_cnt_day'].sum().reset_index())

sns.barplot(x='month',y='item_cnt_day',hue='year',data = grouped)
sns.pointplot(x='month',y='item_cnt_day',hue='year',data=grouped)
#Selecting only relevant features

monthly_sales = train.groupby(['date_block_num','shop_id','item_id'])['date_block_num',

                            'date','item_price','item_cnt_day'].agg({'date_block_num':

                            'mean','date':['min','max'],'item_price':'mean','item_cnt_day':'sum'})

monthly_sales.head()
sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()

#keeping only the test data of valid

sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')

#filling na with zeroes

sales_data_flat.fillna(0,inplace = True)

sales_data_flat.drop(['shop_id','item_id'],inplace = True , axis =1)

sales_data_flat.head()
#Creating the pivot table

pivoted_sales = sales_data_flat.pivot_table(index='ID',columns='date_block_num',fill_value=0,aggfunc = 'sum')

pivoted_sales.head()
X_train = np.expand_dims(pivoted_sales.values[:,:-1],axis=2)

#The last column is our prediction

y_train = pivoted_sales.values[:,-1:]

X_test = np.expand_dims(pivoted_sales.values[:,1:],axis=2)

print(X_train.shape,y_train.shape,X_test.shape)
from keras.layers import Dropout
sales_model = Sequential()

sales_model.add(LSTM(units = 64 , input_shape = (33,1),activation='relu'))

#sales_model.add(LSTM(units = 64, activation = 'relu'))

sales_model.add(Dropout(0.7))

sales_model.add(Dense(1))

sales_model.compile(loss='mse',optimizer = 'adam',metrics=['mean_squared_error'])

sales_model.summary()
sales_model.fit(X_train,y_train,batch_size=4096,epochs=50)
submission_output = sales_model.predict(X_test)

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})

submission.to_csv('submission_stacked.csv',index = False)

submission.head()