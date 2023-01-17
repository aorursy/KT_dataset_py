# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/original.csv")
df
df.shape
df.dtypes
df.describe()
df.corr()
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
High=np.array(df['High'])
Low=np.array(df['Low'])
Close=np.array(df['Close'])
# print(High)
# print("//////////")
# print("Low")
# print(Low)
# print("/////////")
# print("Close")
# print(Close)
High.shape
plt.rcParams["figure.figsize"] = (25,10)
plt.plot(High,color='red',label='HIGH')
plt.plot(Low,color='blue',label='LOW')
plt.plot(Close,color='green',label='CLOSE')
plt.title('Visualisation of High,Low and Close')
plt.legend()
plt.show()
X=df.iloc[:,2:4]
X
X_new=np.array(X)
X_new.shape
Y=df.iloc[:,4:5]
Y
Y_new=np.array(Y)
Y_new.shape
scalar=MinMaxScaler()
scalar.fit(X_new)
X_new=scalar.transform(X_new)
X_new[0:10]
scalar1=MinMaxScaler()
scalar1.fit(Y_new)
Y_new=scalar1.transform(Y_new)
Y_new[0:10]
X_new.shape[0],1
X_new=np.reshape(X_new,(X_new.shape[0],1,X_new.shape[1]))
X_new.shape
X1=X_new[0:200]
Y1=Y_new[0:200]
model=Sequential()

model.add(LSTM(100,activation='tanh',input_shape=(1,2),recurrent_activation='hard_sigmoid'))

model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error',optimizer='rmsprop')
model.fit(X1,Y1,epochs=200,batch_size=25,verbose=2)
Predict=model.predict(X_new[200:250],verbose=1)
print(Predict[0:10])
plt.rcParams["figure.figsize"] = (25,10)
plt.plot(Y_new[200:250], color='green', label='Google Stock Price')
plt.plot(Predict, color='red', label='LSTM Stock Price')
plt.title("GOOGLE STOCK PRICE PREDICTION")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()