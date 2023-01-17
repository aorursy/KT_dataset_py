# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline

import matplotlib.pyplot as plt

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
train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

train.info()
sample=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

test.head()
train.head()
train.drop(["date_block_num"],axis=1,inplace=True)
train["date"]=pd.to_datetime(train["date"],infer_datetime_format=True)
train=train.set_index(['date'])
train=train[['item_cnt_day','shop_id', 'item_id', 'item_price']]
train.head()
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6
train["item_cnt_day"].plot()

plt.xlabel("item_cnt_day")
train.columns
tr=train
j=1

a=['item_cnt_day', 'shop_id', 'item_id', 'item_price']

for i in a:

    tr[i+str(j)]=tr[i].shift(-1)

    
tr
tr.drop(["item_cnt_day","shop_id1","item_id1","item_price1"],axis=1,inplace=True)
tr
tr.fillna(method='ffill',inplace=True)
tr
yy=tr["item_cnt_day1"].values
yy = yy.reshape((len(yy), 1))
tr.drop(["item_cnt_day1"],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler

scale=MinMaxScaler()

scale=scale.fit(tr)

w=scale.transform(tr)
scaler=scale.fit(yy)

qq=scaler.transform(yy)
qq
www=w
xtrain,xtest,ytrain,ytest=www[:2348680,0:3],www[2348680:,0:3],qq[:2348680,-1],qq[2348680:,-1]
xtest
ytest
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)

xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM,Dropout

from keras.layers import Bidirectional
model=Sequential()

model.add(LSTM(100,activation="relu",return_sequences=True,input_shape=(xtrain.shape[1],xtrain.shape[2])))

model.add(Dropout(0.1))

model.add(LSTM(100,activation="relu",return_sequences=True))

model.add(Dropout(0.1))

model.add(LSTM(100,activation="relu",return_sequences=True))

model.add(Dropout(0.1))

model.add(LSTM(100,activation="relu",return_sequences=True))

model.add(Dropout(0.1))

model.add(LSTM(50,activation="relu"))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,batch_size=30000,verbose=2)
ypred=model.predict(xtest)
ypred[:5]
ypred=scaler.inverse_transform(ypred)
ypred
ytest = ytest.reshape((len(ytest), 1))
y_test = scaler.inverse_transform(ytest)
y_test
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,ypred))
y_test[:5]
ypred[:5]
test
train
train=train[['item_price','shop_id', 'item_id']]
trr=train
j=1

a=['item_price','shop_id', 'item_id']

for i in a:

    trr[i+str(j)]=trr[i].shift(-1)

    
trr
trr.drop(["shop_id1","item_id1","item_price"],axis=1,inplace=True)
trr.fillna(method='ffill',inplace=True)
yyy=trr["item_price1"].values

trr.drop(["item_price1"],axis=1,inplace=True)
yyy=yyy.reshape(len(yyy),1)
yyy
trr
from sklearn.preprocessing import MinMaxScaler

scale=MinMaxScaler()

scale=scale.fit(trr)

w=scale.transform(trr)
www=w
yyy
scalerr=scale.fit(yyy)

wewe=scalerr.transform(yyy)
xtrain,xtest,ytrain,ytest=www[:2348680,0:2],www[2348680:,0:2],wewe[:2348680,-1],wewe[2348680:,-1]
xtes=xtest
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)

xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
modelt=Sequential()

modelt.add(LSTM(100,activation="relu",return_sequences=True,input_shape=(xtrain.shape[1],xtrain.shape[2])))

modelt.add(Dropout(0.1))

modelt.add(LSTM(100,activation="relu",return_sequences=True))

modelt.add(Dropout(0.1))

modelt.add(LSTM(100,activation="relu",return_sequences=True))

modelt.add(Dropout(0.1))

modelt.add(LSTM(100,activation="relu",return_sequences=True))

modelt.add(Dropout(0.1))

modelt.add(LSTM(50,activation="relu"))

modelt.add(Dense(1))

modelt.compile(optimizer='adam', loss='mse')
modelt.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,batch_size=30000,verbose=2)
test.head()
test.drop(["ID"],axis=1,inplace=True)
te=test
scalll=scale.fit(test.values)

w=scalll.transform(test.values)
w
w=w.reshape(w.shape[0], w.shape[1], 1)
y_pred=modelt.predict(w)
y_pred
scalerr=scale.fit(yyy)

wewe=scalerr.transform(yyy)
y_pred=scalerr.inverse_transform(y_pred)
y_pred
test
len(y_pred)
test["wewe"]=y_pred
test
scalert=scale.fit(test.values)

wewe=scalert.transform(test.values)
wewe
y_prrr=wewe.reshape(wewe.shape[0], wewe.shape[1], 1)
y_predict=model.predict(y_prrr)
y_predict
scaler=scale.fit(yy)

qq=scaler.transform(yy)
y_predict=scaler.inverse_transform(y_predict)
y_predict
submission=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
submission
pred=y_predict.flatten()
for i in range(214200):

    pred[i]=round(pred[i])
pred
r = pd.Series(pred,name="item_cnt_month")
submiss = pd.concat([pd.Series(range(0,214200),name = "ID"),r],axis = 1)
submiss
submiss.to_csv("11.csv",index=False)