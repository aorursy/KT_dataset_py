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
dataset_train = pd.read_csv("../input/googledta/trainset.csv")
training_set = dataset_train.iloc[:,1:2].values
training_set



# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)



# Creating a data structure with 60 timesteps and t+1 output

X_train = []

y_train = []

for i in range(60, 1258):

    X_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM





# Initialising the RNN

regressor = Sequential()



# Adding the input layer and the LSTM layer

regressor.add(LSTM(units = 3, input_shape = (None, 1)))





# Adding the output layer

regressor.add(Dense(units = 1))

# Compiling the RNN

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Part 3 - Making the predictions and visualising the results



# Getting the real stock price for February 1st 2012 - January 31st 2017

dataset_test = pd.read_csv("../input/googledta/testset.csv")

test_set = dataset_test.iloc[:,1:2].values

real_stock_price = np.concatenate((training_set[0:1258], test_set), axis = 0)















# Getting the predicted stock price of 2017

scaled_real_stock_price = sc.fit_transform(real_stock_price)

inputs = []

for i in range(1258, 1383):

    inputs.append(scaled_real_stock_price[i-60:i, 0])

inputs = np.array(inputs)

inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

predicted_stock_price = regressor.predict(inputs)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)



# Visualising the test set results

import matplotlib.pyplot as plt

plt.plot(test_set, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()



# Visualising the training set with the algorithm prediction on training set

plt.plot(training_set, color = 'red', label = 'Real Google Stock Price')

plt.plot(sc.inverse_transform(regressor.predict(X_train)), color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()



import math

from sklearn.metrics import mean_squared_error

error=math.sqrt(mean_squared_error(predicted_stock_price,test_set))

error
