import numpy as np

import pandas as pd

import matplotlib.pylab as plt

%matplotlib inline



from math import sqrt

import math

from sklearn.metrics import mean_squared_error, mean_absolute_error
# Converting Date(s) to DataTime

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')



# Reading .txt file - ibm.us.txt

# Indexing by Date(s)

data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/ibm.us.txt', sep=',', parse_dates=['Date'], index_col='Date', date_parser=dateparse)



# Printing out the first 5 rows

data.head()
# Checking data types

#data.dtypes



# General statistics about the Dataframe

data.describe()
# Plotting data

plt.figure(figsize=(25,6))

plt.grid(True)

plt.title('IBM Stocks - 1962-2017')

plt.xlabel('Dates')

plt.ylabel('Close Prices ($)')

plt.plot(data['Close'])
# Dividing data

train_data, test_data = data[0:int(len(data)*0.95)], data[int(len(data)*0.95):]



plt.figure(figsize=(25,6))

plt.title('Divinding the data')

plt.grid(True)

plt.ylabel('Close Prices ($)')

plt.plot(data['Close'], 'green', label='Train data')

plt.plot(test_data['Close'], 'blue', label='Test data')

plt.legend()
from matplotlib import pyplot

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error



# Preparing autoregression data

train_ar = train_data['Close']

test_ar = test_data['Close']



# Train autoregression

model = AR(train_ar)

model_fit = model.fit()

window = model_fit.k_ar

coef = model_fit.params



# Walk forward over time steps in test

history = train_ar[len(train_ar)-window:]

history = [history[i] for i in range(len(history))]

predictions = list()

for t in range(len(test_ar)):

    length = len(history)

    lag = [history[i] for i in range(length-window, length)]

    yhat = coef[0]

    for d in range(window):

        yhat += coef[d+1] * lag[window-d-1]

    obs = test_ar[t]

    predictions.append(yhat)

    history.append(obs)



from sklearn.metrics import mean_squared_error, mean_absolute_error

import math



# Mean Squared Error – átlagos négyzetes hiba

mse = mean_squared_error(test_data['Close'], predictions)

print('MSE: ' + str(mse))

# Mean Absolute error

mae = mean_absolute_error(test_data['Close'], predictions)

print('MAE: ' + str(mae))

# Root Mean Squared Error - a MSE négyzetgyöke

rmse = math.sqrt(mean_squared_error(test_data['Close'], predictions))

print('RMSE: ' + str(rmse))
# Plotting autoregression result

plt.figure(figsize=(25,6))

plt.title('Prediction with autoregression')

plt.plot(data.index[-900:], data['Close'].tail(900), color='green', label='Close price')

plt.plot(test_data.index, test_data['Close'], color='red', label='Test close price')

plt.plot(test_data.index, predictions, color='blue', label='Predicted close price')

plt.xticks(rotation=30)

plt.grid(True)

plt.ylabel('Close Prices ($)')

plt.legend()
from pylab import rcParams

rcParams['figure.figsize'] = 25, 8



from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA



data_arima = data['Close']



#import pmdarima

#best_model = pmdarima.auto_arima(train_data['Close'], error_action='ignore', seasonal=True, m=12)

#print("Best model --> (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)







# Best model --> (p, d, q): (2, 1, 2)  and  (P, D, Q, s): (0, 0, 0, 12)
result = seasonal_decompose(data_arima[-1500:], model='multiplicative', freq=365)

fig = result.plot()

plt.show()
train_arima = train_data['Close']

test_arima = test_data['Close']



history = [x for x in train_arima]

y = test_arima



# Make first prediction

predictions = list()

model = ARIMA(history, order=(0, 1, 0))

model_fit = model.fit(disp=0)

yhat = model_fit.forecast()[0]

predictions.append(yhat)

history.append(y[0])



# Rolling forecasts

for i in range(1, len(y)):

    # Predict

    model = ARIMA(history, order=(0, 1, 0))

    model_fit = model.fit(disp=0)

    yhat = model_fit.forecast()[0]



    # Invert transformed prediction

    predictions.append(yhat)



    # Observation

    obs = y[i]

    history.append(obs)



# Report performance

mse = mean_squared_error(y, predictions)

print('MSE: '+ str(mse))

mae = mean_absolute_error(y, predictions)

print('MAE: '+ str(mae))

rmse = math.sqrt(mean_squared_error(y, predictions))

print('RMSE: '+ str(rmse))
# Plotting Arima result

plt.figure(figsize=(25,6))

plt.title('Prediction with Arima')

plt.plot(data.index[-900:], data['Close'].tail(900), color='green', label='Close price')

plt.plot(test_data.index, y, color = 'red', label = 'Real close Price')

plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted close Price')

plt.xticks(rotation=30)

plt.grid(True)

plt.ylabel('Close Prices ($)')

plt.legend()
# LSTM - Long Short Term Memory

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM, GRU

from keras.layers import Dropout



# Convert the dataframe to a numpy array

lstm_data = data.filter(['Close'])

lstm_dataset = lstm_data.values



# Get the number of rows to train the model on

training_data_len = math.ceil(len(lstm_dataset) * 0.95)



# Scale data - preproccesing 

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(lstm_dataset)



scaled_data
# Create the training dataset

# Crate the scaled training data set

train_data = scaled_data[0: training_data_len, :]



# Split the data into x_train and y_train data sets



x_train = [] # Independent training varriables, training features

y_train = [] # Depedent varriables, target varriables



for i in range(60, len(train_data)):

    x_train.append(train_data[i-60:i, 0])

    y_train.append(train_data[i, 0])



# Convert the x_train and y_train to numpy arrays

x_train, y_train = np.array(x_train), np.array(y_train)

#x_train.shape



# Reshape the data - LSTM expects the input to be 3D (Currently: 2D)

# Number of samples = number of rows

# Number of time steps = 60

# Number of features = 1 (Close price)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Features 
# Build the LSTM model

model = Sequential()



# Layers

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(25))

model.add(Dense(1))



# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')



# Train/fit the model

model.fit(x_train, y_train, batch_size=1, epochs=10)
# Creating the testing data

test_data = scaled_data[training_data_len - 60: , :]



# Creating the data sets: x_test and y_test

x_test = []

y_test = lstm_dataset[training_data_len:, :]

for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i, 0])



# Convert the data to a numpy array

x_test = np.array(x_test)

#x_test.shape



# Reshape the data

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 



#  Get the models predicted price values

predictions = model.predict(x_test)

print(type(predictions))

predictions = scaler.inverse_transform(predictions) # Unscaling the data
# Report performance



# MSE - Mean Squared Error

mse = mean_squared_error(y_test, predictions)

print('MSE: '+ str(mse))

# MSE - Mean Absolute Error

mae = mean_absolute_error(y_test, predictions)

print('MAE: '+ str(mae))

# RMSE - Root Mean Squared Error

rmse = math.sqrt(mean_squared_error(y_test, predictions))

print('RMSE: '+ str(rmse))
# Plot the result

test_data = data[math.ceil(len(data)*0.95):]



plt.figure(figsize=(25,6))

plt.title('Prediction with LSTM')

plt.plot(data.index[-900:], data['Close'].tail(900), color='green', label='Close price')

plt.plot(test_data.index, test_data['Close'], color='red', label='Test close price')

plt.plot(test_data.index, predictions, color='blue', label='Predicted close price')

plt.xticks(rotation=30)

plt.grid(True)

plt.ylabel('Close Prices ($)')

plt.legend()