import numpy as np

import pandas as pd

import os
os.listdir('../input')
sales_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
sales_data.info()
sales_data.describe()
sales_data.isnull().sum()
sales_data.isna().sum()
print(test_data.info())

print('*'*100)

print(test_data.describe())

print('*'*100)

print(test_data.isnull().sum())

print('*'*100)

print(test_data.isna().sum())
print(item_cat.info())

print('*'*100)

print(item_cat.describe())

print('*'*100)

print(item_cat.isnull().sum())

print('*'*100)

print(item_cat.isna().sum())
print(items.info())

print('*'*100)

print(items.describe())

print('*'*100)

print(items.isnull().sum())

print('*'*100)

print(items.isna().sum())
sales_data.dtypes
sales_data['date'] = pd.to_datetime(sales_data['date'],format='%d.%m.%Y')
dFrame = sales_data.pivot_table(index=['shop_id','item_id'],values=['item_cnt_day'],columns = ['date_block_num'],fill_value=0,aggfunc=sum)
dFrame.reset_index(inplace=True)
dFrame.head()
dFrame = pd.merge(test_data,dFrame,on = ['item_id','shop_id'],how='left')
dFrame.fillna(0,inplace=True)
dFrame.head()
dFrame.drop(['ID','shop_id','item_id'],inplace=True,axis=1)

dFrame.head()


Xtrain = np.expand_dims(dFrame.values[:,:-1],axis = 2)

Ytrain = dFrame.values[:,-1:]

Xtest = np.expand_dims(dFrame.values[:,1:],axis = 2)
def sepraterPattern(i,n):

    if(i==1):

        print('*'*n)

    else:

        print('-'*n)
print("X Train : ", Xtrain.shape)

sepraterPattern(1,30)

print("Y Train : ", Ytrain.shape)

sepraterPattern(1,30)

print("X Test : ", Xtest.shape)

sepraterPattern(1,30)
from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout

model  = Sequential()

model.add(LSTM(units=64,input_shape=(33,1)))

model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(loss = 'mse',optimizer = 'SGD', metrics = ['mean_squared_error'])
model.summary()
model.fit(Xtrain,Ytrain,batch_size = 4096,epochs = 15)
sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample_submission.describe()
sample_submission.head()
submissionModel = model.predict(Xtest)
submissionModel = submissionModel.clip(0,20)
# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submissionModel.ravel()})

# creating csv file from dataframe

submission.to_csv('submissionFiles.csv',index = False)