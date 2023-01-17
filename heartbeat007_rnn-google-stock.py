!pip install tensorflow-gpu
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
## import the data from the quandl

!pip install quandl

import quandl
quandl.ApiConfig.api_key = 'zuiQMfguw3rRgLvkCzxk'
df = quandl.get('WIKI/GOOGL')
df.head()
df.tail()
##checking is there null value

df.isnull().sum()
df.corr()[['Adj. Close']]
## dropping the split ratio adj.Vlume Volume EX-Divident

df=df.drop('Volume',axis=1)
df=df.drop('Adj. Volume',axis=1)

df=df.drop('Split Ratio',axis=1)

df=df.drop('Ex-Dividend',axis=1)
df.head()
df.corr()['Adj. Close'].plot()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
X = df.drop('Adj. Close',axis=True)

y = df[['Adj. Close']]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)

result = y_test

scaler = MinMaxScaler()

scaler.fit(x_train)

xtrain_t = scaler.transform(x_train)

scaler.fit(x_test)

xtest_t = scaler.transform(x_test)

scaler.fit(y_train)

y_train =scaler.transform(y_train) # we transform the y so after predict we have to inverse transeform it

scaler.fit(y_test)

y_test =scaler.transform(y_test) # we transform the y so after predict we have to inverse transeform it
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import Flatten
x_train = np.reshape(xtrain_t, (xtrain_t.shape[0],xtrain_t.shape[1],1))

x_test = np.reshape(xtest_t, (xtest_t.shape[0],xtest_t.shape[1],1))
print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
regressor = Sequential()

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['accuracy'])
regressor.fit(x_train,y_train,epochs = 100)
y_pred = regressor.predict(x_test)
output = scaler.inverse_transform(y_pred)
real_output = []

for item in output:

  real_output.append((item[0]))
actual_output = []

for item in result['Adj. Close']:

  actual_output.append((item))
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual_output,real_output)
mse
result['predited value'] = np.array(real_output)
result
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,20)

result.plot()