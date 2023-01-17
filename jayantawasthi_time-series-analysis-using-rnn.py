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
train=pd.read_csv("/kaggle/input/time-series-datasets/Electric_Production.csv")
len(train)
train.head()
train1=train["IPG2211A2N"]
plt.plot(train1)
train1=train1.values.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler(feature_range=(0,1))
scale=scale.fit(train1)

scale.data_min_
normalize=scale.transform(train1)
normalize[:5]
res=normalize.flatten()
res[:5]
tt={}

ww=[]
uu=[]
def make(a,b):
        m=0
        k=0
        while k<394:
            qq=[]
            for i in range(b):
                        u=a[m]
                        m=m+1
                        qq.append(u)
            ww.append(qq)  
            uu.append(a[m])
   
            m=k+1
            k=k+1

   
        
make(res,3)
ww[:5]
from numpy import array
www=array(ww)
www[:5]
uu=array(uu)
xtrain,ytrain=www[:325],uu[:325]
xtest,ytest=www[325:],uu[325:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(Bidirectional(LSTM(50,activation="relu",input_shape=(394,1))))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,batch_size=40,verbose=0)
y_pred=model.predict(xtest)
y_pred[:5]
ypred=scale.inverse_transform(y_pred)
ytest=ytest.reshape(-1,1)
ytest=scale.inverse_transform(ytest)
ypred[:5]
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(ytest,ypred))
train1=pd.read_csv("/kaggle/input/time-series-datasets/monthly-beer-production-in-austr.csv")

train1.head()
trainx=train1["Monthly beer production"]
plt.plot(trainx)
trainx=trainx.values.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler(feature_range=(0,1))
scale=scale.fit(trainx)
scale.data_min_
normalize=scale.transform(trainx)
normalize[:5]
res=normalize.flatten()
res[:5]
len(trainx)
len(res)
tt={}

ww=[]
uu=[]
def make(a,b):
        m=0
        k=0
        mm=476-b
        while k<mm:
            qq=[]
            for i in range(b):
                        u=a[m]
                        m=m+1
                        qq.append(u)
            ww.append(qq)  
            uu.append(a[m])
   
            m=k+1
            k=k+1

make(res,3)
ww[:5]
uu=array(uu)
www=array(ww)
xtrain,ytrain=www[:325],uu[:325]
xtest,ytest=www[325:],uu[325:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Bidirectional
model=Sequential()
model.add(LSTM(50,activation="relu",input_shape=(473,1)))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,verbose=2)
y_pred=model.predict(xtest)
y_pred[:5]
ypred=scale.inverse_transform(y_pred)
ytest=ytest.reshape(-1,1)
ytest=scale.inverse_transform(ytest)
ypred[:5]
ytest[:5]
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(ytest,ypred))