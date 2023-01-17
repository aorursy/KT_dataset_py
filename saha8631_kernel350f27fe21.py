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
train_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')


test_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
train_data
Sample_submission_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
Sample_submission_data
test_data
test_data
item_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_data
train_data=pd.merge(train_data,item_data,on='item_id',how='inner')
train_data
item_categories_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories_data



train_data=pd.merge(train_data,item_categories_data,on='item_category_id',how='inner')
train_data.item_cnt_day


shops_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
shops_data
train_data=pd.merge(train_data,shops_data,how='inner',on='shop_id')
train_data
train_data=train_data.sort_values('date',ascending=True)
train_data
g=train_data.groupby('date')
Item_count_per_day=g['item_cnt_day'].sum()
train_data['item_cnt_day'].value_counts()
len(train_data[train_data['item_cnt_day']<0])
train_data['item_cnt_day']=train_data['item_cnt_day'].mask(train_data['item_cnt_day'].lt(0),train_data['item_cnt_day'].median())
train_data['item_price']=train_data['item_price'].mask(train_data['item_price'].lt(0),train_data['item_price'].median())
import matplotlib.pyplot as plt

plt.subplots(figsize=(20,20))

Item_count_per_day.plot()
train_data
import matplotlib.pyplot as plt

item_sales_count=train_data.groupby('date_block_num')['item_cnt_day'].sum()

plt.subplots(figsize=(30,10))

item_sales_count.plot()
f=train_data.groupby('shop_id')['item_id'].count()

f
k=train_data.groupby('item_id')['item_category_id'].count()

k
train_data.isnull().sum()/len(train_data)
train_data['item_id'].value_counts()
train_data.columns
train_data['item_name'].value_counts()
train_data['shop_id'].value_counts()
train_data['shop_name'].value_counts()
del train_data['item_name']

del train_data['item_category_name']

del train_data['shop_name']
train_data
train_data=train_data.sort_values('date',ascending=True)
train_data
train_data
c=train_data.groupby('date_block_num')['item_cnt_day'].sum()
train_data
import seaborn as sns

sns.boxplot(x='item_price',data=train_data)
#Outlier Detection

for i in range(0,100,10):

    var =train_data["item_price"].values

    var = np.sort(var,axis = None)

    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))

print("100 percentile value is ",var[-1])
for i in range(90,100):

    var =train_data['item_price'].values

    var = np.sort(var,axis = None)

    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))

print("100 percentile value is ",var[-1])
for i in np.arange(0.0,1.0,0.1):

    var=train_data['item_price'].values

    var=np.sort(var,axis=None)

    print(99+i,var[int(len(var)*float(99+i)/100)])

print(99+i+0.1,var[-1]) 
train_data=train_data[train_data['item_price']<23990.0]


train_data
train_data['date_block_num'].value_counts()
train_data.columns
l=train_data.groupby(['date_block_num','shop_id','item_id'])['item_cnt_day'].sum()
train_data_final=train_data.join(train_data.groupby(['date_block_num','shop_id','item_id'])['item_cnt_day'].sum(),on=['date_block_num','shop_id','item_id'],rsuffix='_r')
train_data_final
b=train_data_final['item_cnt_day_r']

train_data_final['item_cnt_month']=b
del train_data_final['item_cnt_day_r']
train_data_final.columns
del train_data_final['item_category_id']
del train_data_final['item_cnt_day']
del train_data_final['item_price']
train_data_final
train_data_final=train_data_final.sort_values('date',ascending =True)
train_data_final
#Date Feature Extraction 

train_data_final['date']=pd.to_datetime(train_data_final['date'])
train_data_final.info()
train_data_final.columns
#Time_based_splitiing

X=train_data_final.drop('item_cnt_month',axis=1)

y=train_data_final['item_cnt_month']

from sklearn.model_selection import TimeSeriesSplit

tscv=TimeSeriesSplit(n_splits=3)

for train_index,test_index in tscv.split(X):

    X_train,X_test=X[ :len(train_index)],X[len(train_index):(len(train_index)+len(test_index))]

    y_train,y_test=y[ :len(train_index)],y[len(train_index):(len(train_index)+len(test_index))]

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



#Featurize the data

X_train.columns
del X_train['date']

del X_test['date']
print(X_train.shape)

print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range=(0.1,1.1))

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
y_train=y_train.values

y_test=y_test.values
y_test=y_test.reshape(-1,1)

y_train=y_train.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler

sc1=MinMaxScaler(feature_range=(0.1,1.1))

y_train=sc1.fit_transform(y_train)

y_test=sc1.fit_transform(y_test)
X_train.shape
X_train_final=X_train.reshape(X_train.shape[0],1,X_train.shape[1])

X_test_final=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
X_train_final.shape
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout,LSTM

#from keras.optimizers import RMSProp

from keras.constraints import max_norm

from keras.initializers import RandomNormal



nb_epochs=3
model=Sequential()
model.add(LSTM(40,return_sequences=True,activation='relu',input_shape=(X_train_final.shape[1],3,),kernel_initializer=RandomNormal(mean=0.0,stddev=0.2236,seed=1)))

model.add(Dropout(0.4))
#model.add(LSTM(50,return_sequences=True,activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.2,seed=1)))

#model.add(Dropout(0.5))
model.add(LSTM(20,return_sequences=False,activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.3162,seed=1)))

model.add(Dropout(0.4))
#model.add(LSTM(10,return_sequences=False,activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.4472,seed=1)))

#model.add(Dropout(0.5))
#model.add(LSTM(10,return_sequences=False,activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.4472,seed=None)))

#model.add(Dropout(0.5))
model.add(Dense(units=1,activation='relu'))
model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['mse'])
history=model.fit(X_train_final,y_train,epochs=nb_epochs,batch_size=1024,validation_data=(X_test_final,y_test),verbose=1)
def plt_dynamic(x,vy,ty,ax,colors=['b']):

    ax.plot(x,vy,'b',label='Validation Loss')

    ax.plot(x,ty,'r',label='Train loss')

    plt.legend()

    plt.grid()

    fig.canvas.draw()
import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,1)

ax.set_xlabel('Epoch')

ax.set_ylabel('Validation_loss')

x=list(range(1,nb_epochs+1))

vy=history.history['val_loss']

ty=history.history['loss']

plt_dynamic(x,vy,ty,ax)

final_test_mse=history.history['val_mse'][-1]
from math import sqrt

RMSE=sqrt(final_test_mse)

print(RMSE)
trainPredict=model.predict(X_train_final)

testPredict=model.predict(X_test_final)
trainPredict_in = sc1.inverse_transform(trainPredict)

trainY = sc1.inverse_transform(y_train)

testPredict_in = sc1.inverse_transform(testPredict)

testY = sc1.inverse_transform(y_test)
from sklearn.metrics import mean_squared_error

from math import sqrt

trainScore = sqrt(mean_squared_error(trainY[0], trainPredict_in[0]))

testScore = sqrt(mean_squared_error(testY[0], testPredict_in[0]))
testScore
trainScore
test_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
test_data

test_data['date_block_num']=34
r=test_data['ID']
del test_data['ID']
test_data
X_test=test_data
#Featurizing the test data





X_test=sc.fit_transform(X_test)



X_test_scaled=X_test.reshape((X_test.shape[0],1,X_test.shape[1]))
X_test_scaled.shape
y_predict=model.predict(X_test_scaled)
y_predict
y_final_predict=sc1.inverse_transform(y_predict)
y_final_predict.shape
test_data['item_cnt_month']=y_final_predict
test_data
test_data['ID']=r
test_data
del test_data['date_block_num']
test_data
test_data_fl=test_data.drop(['item_id','shop_id'],axis=1)
Final_data=test_data_fl[['ID','item_cnt_month']]
Final_data.to_csv('LSTM.csv', index=False)