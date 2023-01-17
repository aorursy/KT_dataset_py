# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the training set

dataset_train = pd.read_csv('../input/Google_Stock_Price_Train.csv')
dataset_train
dataset_train.head()
train = dataset_train.loc[:, ["Open"]].values

train
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

train_scaled = scaler.fit_transform(train)

train_scaled
plt.plot(train_scaled)

plt.show()
x_train = []

y_train = []

timesteps = 50 

for i in range(timesteps,1250):

    x_train.append(train_scaled[i-timesteps:i,0])

    y_train.append(train_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
#Reshaping

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

x_train

    
y_train
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout

# Initialising the RNN

regressor = Sequential()



# Adding the first RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30,activation='tanh', return_sequences = True, input_shape = (x_train.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30,activation='tanh', return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30,activation='tanh', return_sequences = True))

regressor.add(Dropout(0.2))





# Adding a fourth RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(x_train, y_train, epochs = 250, batch_size = 629)
# Getting the real stock price of 2017

dataset_test = pd.read_csv('../input/Google_Stock_Price_Test.csv')

dataset_test.head()
real_stock_price = dataset_test.loc[:, ["Open"]].values

real_stock_price
# Getting the predicted stock price of 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs)  # min max scaler

inputs
x_test = []

for i in range(timesteps, 70):

    x_test.append(inputs[i-timesteps:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)



# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()
#Lets try them different optimizer --> RMSProp



# Initialising the RNN

regressor = Sequential()



# Adding the first RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30,activation='tanh', return_sequences = True, input_shape = (x_train.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30,activation='tanh', return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30,activation='tanh', return_sequences = True))

regressor.add(Dropout(0.2))





# Adding a fourth RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 30))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'RMSProp', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(x_train, y_train, epochs = 250, batch_size = 629)
# Getting the real stock price of 2017

dataset_test = pd.read_csv('../input/Google_Stock_Price_Test.csv')

dataset_test.head()
real_stock_price = dataset_test.loc[:, ["Open"]].values

real_stock_price
# Getting the predicted stock price of 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs)  # min max scaler

inputs
x_test = []

for i in range(timesteps, 70):

    x_test.append(inputs[i-timesteps:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)



# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()