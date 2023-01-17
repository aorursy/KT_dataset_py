! pip install tensorflow-gpu
import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

df = pd.read_csv("../input/aple.us.txt")
df.head()
df.isnull().sum()
df.OpenInt.value_counts()
df = df.drop('OpenInt',axis=1)
df.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
df=df.set_index('Date')

X =df.drop('Close',axis=1)

y = df[['Close']]

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

from keras import Sequential

from keras.layers import Dense,Dropout,LSTM,Flatten




print (x_train.shape)

print (x_test.shape)
##have to convert to 3 dim for feeding RNN

x_train = np.array(x_train)

x_test = np.array(x_test)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
regressor = Sequential()

regressor.add(LSTM(units = 50,return_sequences = True))

regressor.add(Dropout(.2))

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1)) #we want single feature output which is df['Close']

regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(x_train,y_train,epochs = 100)
y_pred = regressor.predict(x_test)
y_pred
## we have to inverse transform it cause we transform the x_test before

output = scaler.inverse_transform(y_pred)
output
real_output = []

for item in output:

    real_output.append((item[0]))

result['predited value'] = np.array(real_output)
result.head()
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,20)

result.plot()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(np.array(result['Close']), np.array(result['predited value']))
import math

print ("MSE: "+str(mse))

print ("MSE: "+str(math.sqrt(mse)))