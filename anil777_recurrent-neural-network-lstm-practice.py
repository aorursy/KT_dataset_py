import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')
def plot_predictions(test,predicted):

    plt.plot(test, color='red',label='Real IBM Stock Price')

    plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')

    plt.title('IBM Stock Price Prediction')

    plt.xlabel('Time')

    plt.ylabel('IBM Stock Price')

    plt.legend()

    plt.show()



def return_rmse(test,predicted):

    rmse = math.sqrt(mean_squared_error(test, predicted))

    print("The root mean squared error is {}.".format(rmse))
dataset = pd.read_csv('../input/IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

dataset.head()
# Checking for missing values

training_set = dataset[:'2016'].iloc[:,1:2].values

test_set = dataset['2017':].iloc[:,1:2].values
dataset["High"][:'2016'].plot(figsize=(16,4),legend=True)

dataset["High"]['2017':].plot(figsize=(16,4),legend=True)

plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])

plt.title('IBM stock price')

plt.show()
sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)
# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output

# So for each element of training set, we have 60 previous training set elements 

X_train = []

y_train = []

for i in range(60,2769):

    X_train.append(training_set_scaled[i-60:i,0])

    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
print(y_train)
# Reshaping X_train for efficient modelling



X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

X_train.shape
# LSTM architecture 



regressor = Sequential()



regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units=50))

regressor.add(Dropout(0.2))



regressor.add(Dense(units=1))



regressor.compile(optimizer='rmsprop',loss='mean_squared_error')

# Fitting to the training set

regressor.fit(X_train,y_train,epochs=50,batch_size=32)
# Now to get the test set ready in a similar way as the training set.

# The following has been done so forst 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 

# 'High' attribute data for processing

dataset_total = pd.concat((dataset["High"][:'2016'],dataset["High"]['2017':]),axis=0)

inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values

inputs = inputs.reshape(-1,1)

inputs  = sc.transform(inputs)
# Preparing X_test and predicting the prices

X_test = []

for i in range(60,311):

    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plot_predictions(test_set,predicted_stock_price)
# Evaluating our model

return_rmse(test_set,predicted_stock_price)
regressorGRU = Sequential()



regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))



regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))





regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))



regressorGRU.add(GRU(units=50, activation='tanh'))

regressorGRU.add(Dropout(0.2))



regressorGRU.add(Dense(units=1))



regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')

# Fitting to the training set

regressorGRU.fit(X_train,y_train,epochs=50,batch_size=150)

# Preparing X_test and predicting the prices

X_test = []

for i in range(60,311):

    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

GRU_predicted_stock_price = regressorGRU.predict(X_test)

GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)
plot_predictions(test_set,GRU_predicted_stock_price)