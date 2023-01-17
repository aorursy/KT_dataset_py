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
train=pd.read_csv("/kaggle/input/stock-time-series-20050101-to-20171231/CAT_2006-01-01_to_2018-01-01.csv")
train.tail()
train.drop(["Name"],axis=1,inplace=True)
train
train["Date"]=pd.to_datetime(train["Date"],infer_datetime_format=True)
train=train.set_index(['Date'])
train.head()
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6

train["Open"].plot()

plt.xlabel("open")
train["High"].plot()

plt.xlabel("High")
train["Low"].plot()

plt.xlabel("Low")
train["Close"].plot()

plt.xlabel("Close")
train["Volume"].plot()

plt.xlabel("Volume")
from sklearn.preprocessing import MinMaxScaler

scale=MinMaxScaler()

scale=scale.fit(train)

data=scale.transform(train)
def sss(w):

    t1,t2=w.shape

    a=[]

    t={}

    for i in range(t2):

        for j in range(t1):

                a.append(w[j][i])

        ss=pd.Series(a)

        t.update({i:ss})

        a=[]

    k=[]

    for i,j in t.items():

        k.append(j)

    ww=pd.concat(k,axis=1)

    n=len(ww.columns)

    for i in range(n):

         ww[n+i]=ww[i].shift(-1)

    return ww

    
w=sss(data)
w
w.fillna(method='ffill',inplace=True)
w.drop([6,7,8,9],axis=1,inplace=True)

w
w.info()
www=w.values
xtrain,xtest,ytrain,ytest=www[:2015,0:5],www[2015:,0:5],www[:2015,-1],www[2015:,-1]
xtes=xtest
xtes
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)

xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM,Dropout

from keras.layers import Bidirectional
model=Sequential()

model.add(Bidirectional(LSTM(50,activation="relu",return_sequences=True,input_shape=(xtrain.shape[1],xtrain.shape[2]))))

model.add(LSTM(50,activation="relu"))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,batch_size=32,verbose=2)
y_pred=model.predict(xtest)
y_pred[:5]
from numpy import concatenate
y_pred= concatenate((y_pred,xtes[:, 1:]), axis=1)
xtes[:5]
y_pred[:5]
y_pred = scale.inverse_transform(y_pred)
ypred= y_pred[:,0]

y_pred[:5]
ytest = ytest.reshape((len(ytest), 1))
ytest[:5]


from numpy import concatenate
ytest= concatenate((ytest,xtes[:, 1:]), axis=1)
y_test = scale.inverse_transform(ytest)
y_test=y_test[:,0]

y_test[:5]
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,ypred))
y_test[:5]
ypred[:5]