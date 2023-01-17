#Importing Linraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

import keras
#Reading dataset

data_set = pd.read_csv('../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv', na_values='ND')
data_set.shape
#Dataset head

data_set.head()
#Checking null values

data_set.isnull().sum()
data_set.interpolate(inplace=True)

data_set.isnull().sum()
#Plotting Indian Exchange rate

plt.plot(data_set['INDIA - INDIAN RUPEE/US$'])
#Data frame

df = data_set['INDIA - INDIAN RUPEE/US$']
print(df)
#Preprocessing data set

df = np.array(df).reshape(-1,1)
df
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df = scaler.fit_transform(df)
df
#Training and test sets

train = df[:4800]

test = df[4800:]



print(train.shape)

print(test.shape)
def get_data(data, look_back):

  datax, datay = [],[]

  for i in range(len(data)-look_back-1):

    datax.append(data[i:(i+look_back),0])

    datay.append(data[i+look_back,0])

  return np.array(datax) , np.array(datay)
look_back = 1



x_train , y_train = get_data(train, look_back)

print(x_train.shape)

print(y_train.shape)
x_test , y_test = get_data(test,look_back)

print(x_test.shape)

print(y_test.shape)
#Processing train and test sets for LSTM model

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)

x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
print(x_train.shape)

print(x_test.shape)

#Defining the LSTM model

from keras.models import Sequential

from keras.layers import Dense,LSTM

n_features=x_train.shape[1]

model=Sequential()

model.add(LSTM(100,activation='relu',input_shape=(1,1)))

model.add(Dense(n_features))
model.summary()
#Compiling

model.compile(optimizer='adam', loss = 'mse')
#Training

model.fit(x_train,y_train, epochs = 5, batch_size=1)
#Prediction using the trained model

scaler.scale_
y_pred = model.predict(x_test)

y_pred = scaler.inverse_transform(y_pred)

print(y_pred[:10])
#Processing test shape

y_test = np.array(y_test).reshape(-1,1)

y_test = scaler.inverse_transform(y_test)

print(y_test[:10])
#Visualizing the results

plt.figure(figsize=(10,5))

plt.title('Foreign Exchange Rate of India')

plt.plot(y_test , label = 'Actual', color = 'g')

plt.plot(y_pred , label = 'Predicted', color = 'r')

plt.legend()
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)