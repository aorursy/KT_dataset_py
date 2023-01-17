import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

from datetime import datetime, date

from keras.regularizers import l2

from keras.models import Model, Sequential

from keras.layers import *
path = "/kaggle/input/competitive-data-science-predict-future-sales/"

sales_train = pd.read_csv(path + 'sales_train.csv')

test = pd.read_csv(path + 'test.csv')

submission = pd.read_csv(path + 'sample_submission.csv')

items = pd.read_csv(path + 'items.csv')

item_cats = pd.read_csv(path + 'item_categories.csv')

shops = pd.read_csv(path + 'shops.csv')
print(sales_train.info())
sales_train['date'] = pd.to_datetime(sales_train['date'],format = '%d.%m.%Y')
train = sales_train.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})

train.reset_index(inplace=True)

train.columns = ['date_block_num', 'shop_id', 'item_id','item_cnt_month']

#stay only with the last two years->more relevant to us.

train = (train[( (train['date_block_num'] >= 8) & 

                          (train['date_block_num'] <= 33) )]).reset_index().drop('index',axis=1)

train.reset_index(inplace=True)



train = train.query('item_cnt_month >=0 and item_cnt_month <=200')
shopItemMonth = train.pivot_table(index=['shop_id','item_id'], columns='date_block_num',values='item_cnt_month', fill_value=0).reset_index()
dataset = pd.merge(test,shopItemMonth, on = ['item_id','shop_id'], how='left')
dataset.fillna(0,inplace=True)

dataset.drop(['shop_id','item_id','ID'], inplace=True, axis=1)

dataset.describe().T
dataset
X_train = np.expand_dims(dataset.values[:,:-1],axis=2)

y_train = dataset.values[:,-1:]



X_test = np.expand_dims(dataset.values[:,1:],axis=2)

y_train
plt.figure(figsize=(17,5))

plt.plot(X_train[:,0])

plt.plot(y_train[30000:,0],c='red',alpha=0.3)
model = Sequential()

model.add(LSTM(48, input_shape=(25,1)))           #lookback 2 years

model.add(Dropout(0.4))

model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

model.summary()
model.fit(X_train,y_train,batch_size=4096,epochs=50)
# creating submission file 

submission_pfs = model.predict(X_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs17.csv',index = False)
!cp sub_pfs17.csv drive/My\ Drive/Assignments/Ass2