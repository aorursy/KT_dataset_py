# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importing libraries



import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")



from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional

from keras.optimizers import SGD



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



import math



import warnings

warnings.filterwarnings("ignore")
# function which plots ibm stock prices: real and predicted both



def plot_predictions(test, predicted):

    plt.plot(test, color="red", label="real IBM stock price")

    plt.plot(predicted, color="blue", label="predicted stock price")

    plt.title("IBM stock price prediction")

    plt.xlabel("time")

    plt.ylabel("IBM stock price")

    plt.legend()

    plt.show()
# function which calculates root mean squared error



def return_rmse(test, predicted):

    rmse = math.sqrt(mean_squared_error(test, predicted))

    print("the root mean squared error is : {}.".format(rmse))
data = pd.read_csv("../input/IBM_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=["Date"])



data.shape
data.head(5)
train = data[:'2016'].iloc[:,1:2].values

test = data['2017':].iloc[:,1:2].values
# visualization of "High" attribute of the dataset



data["High"][:'2016'].plot(figsize=(16,4), legend=True)

data["High"]["2017":].plot(figsize=(16,4), legend=True)

plt.legend(["Training set (before 2017)", "Test set (from 2017)"])

plt.title("IBM stock prices")

plt.show()
# scaling the training set



sc = MinMaxScaler(feature_range=(0,1))

train_scaled = sc.fit_transform(train)
# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output

# So for each element of training set, we have 60 previous training set elements



x_train = []

y_train = []



for i in range(60,2769):

    x_train.append(train_scaled[i-60:i, 0])

    y_train.append(train_scaled[i,0])



x_train, y_train = np.array(x_train), np.array(y_train)
x_train[0]
y_train[0]
len(x_train)
len(y_train)
x_train.shape
y_train.shape
# reshaping x_train for efficient modelling



x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape
# LSTM architecture



regressor = Sequential()



# add first layer with dropout



regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))

regressor.add(Dropout(0.2))



# add second layer



regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))



# add third layer



regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))



# add fourth layer



regressor.add(LSTM(units=50))

regressor.add(Dropout(0.2))



# the output layer



regressor.add(Dense(units=1))
# compiling the LSTM RNN network



regressor.compile(optimizer='rmsprop', loss='mean_squared_error')



# fit to the training set



regressor.fit(x_train, y_train, epochs=5, batch_size=32)
# Now to get the test set ready in a similar way as the training set.

# The following has been done so forst 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 'High' attribute data for processing



dataset_total = pd.concat((data['High'][:'2016'], data['High']['2017':]), axis=0)

print(dataset_total.shape)



inputs = dataset_total[len(dataset_total)-len(test)-60 : ].values

print(inputs.shape)

inputs = inputs.reshape(-1,1)

print(inputs.shape)

inputs = sc.transform(inputs)

print(inputs.shape)
# preparing x_test



x_test = []

for i in range(60,311):

    x_test.append(inputs[i-60:i, 0])

    

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# predicting the stock prices for test set



predicted = regressor.predict(x_test)

predicted = sc.inverse_transform(predicted)
# visualizing the results: predicted vs test



plot_predictions(test, predicted)
# evaluating the model



return_rmse(test, predicted)
# The GRU architecture

regressorGRU = Sequential()

# First GRU layer with Dropout regularisation

regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))

# Second GRU layer

regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))

# Third GRU layer

regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))

# Fourth GRU layer

regressorGRU.add(GRU(units=50, activation='tanh'))

regressorGRU.add(Dropout(0.2))

# The output layer

regressorGRU.add(Dense(units=1))
# compiling the model



regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')



# fitting the model



regressorGRU.fit(x_train, y_train, epochs=5, batch_size=150)
# predicting the stock prices for test set and visualization



predicted_with_gru = regressorGRU.predict(x_test)

predicted_with_gru = sc.inverse_transform(predicted_with_gru)



plot_predictions(test, predicted_with_gru)
# evaluating the model performance



return_rmse(test, predicted_with_gru)