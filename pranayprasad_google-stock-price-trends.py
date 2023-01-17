import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset_train = pd.read_csv('../input/google-stock-price/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
training_set.shape
# Normalizing the inputs instead of Standardizing

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
for i in range(120, 1258):                               # Starting from 120 as we can analyse only after first 120 days. 1258 - Index of the last date
    X_train.append(training_set_scaled[i-120:i, 0])      # So it goes like 0-120 ,1-121,2-122....1198-1138
    y_train.append(training_set_scaled[i, 0])            # Just the 121st observation 
X_train, y_train = np.array(X_train), np.array(y_train)  # Converting to numpy arrays
X_train.shape
# So it has 1138 rows and 120 columns

X_train[0:2] 
y_train.shape
# y_train has 1138 rows and 1 column
y_train[0:10]
# RNN expects input in the form of a 3D tensor with shape [batch(rows), timesteps(columns), feature(1 if only 1 predictor, you can add more predictors here)]
# Refer Keras docs : https://keras.io/api/layers/recurrent_layers/lstm/ and see the 'input'

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
X_train.shape
import tensorflow as tf
tf.__version__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout # To prevent overfitting
# Initializing RNN as a sequence of layers as opposed of Computation Graphs. 
# Note to self - I will be using Computational Graphs (Dynamic Graphs) using Pytorch in future as it is much more powerful.

regressor = Sequential()
# units = no of cells in this layer. Since predicting the stock price is a pretty complex problem we need a pretty high 'units'.

# return_sequences = Signifies if there is another LSTM layer after this (True / False(default)). 
#                    If there is no LSTM layer after current layer don't mention this arg as by default the value is 'False'
#                    If we have multiple LSTM's in our network it is also called 'Stacked LSTMS'

# input_shape = Exactly same as Reshaping . Just mention the last 2 dimensions i.e timesteps and features as the first (batch) is automatically taken in account.

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # To prevent overfitting - 0.2 is just a classic value. You can tweak it.
# No need to specify any input_shape this time.

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Here return_sequences = 'False' as there is no LSTM layer after that. 

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Since it is a regression problem the output layer will always be a single neuron.

regressor.add(Dense(units = 1))
# loss = 'mean_squared_error' as this is a regression problem.

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Note - The data provided is not a lot. It is just a 5 years data. If maybe i can go to Yahoo Finance and scrape some data i can get more data and make better model.

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 
dataset_test = pd.read_csv('../input/google-stock-price/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
# This is the REAL Google stock price for Jan 2017 we will compare this with predicted stock price.
real_stock_price
# To predict we first need inputs of last 120 days data everyday of the month Jan 2017. This data stretches to both training and test data. 
# For that we need to concatinate both the training set and the test set data.
# However we will not concat the training_set and real_stock_price as we should never touch the actual test set.
# So we will concat the dataframes on vertical axis (axis = 0)

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
dataset_total
# We need the inputs of past 120 days of every day of Jan 17.
# The lower bound of the input will be first working day of Jan - 120. First working day of Jan = len(dataset_total) - len(dataset_test)
# Upper bound will be the last value of Jan 17

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120 :].values
inputs
inputs.shape
# Formatting the input . Since we didn't use iloc we didn't get the normal format.

inputs = inputs.reshape(-1,1)
inputs
inputs.shape
# Scaling the inputs using the same scaler

inputs = sc.transform(inputs)
# Making X_test the same way made X_train in data preprocessing by making a data structure with 120 timestamps.

X_test = []

for i in range(120, 140):                    # 140 - Index of the last date of 'inputs'
    X_test.append(inputs[i-120:i, 0])        # So it goes like 0-120 ,1-121,2-122....20-140

X_test = np.array(X_test) # Converting to numpy arrays
X_test.shape
# Converting X_test to a 3D tensor format (same as we did for X_train)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape
# THE PREDICT STEP

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Checking the predicted values

predicted_stock_price
# Again we have to compare it with real_stock_price.
# Note as i said before we can't predict the stock prices but we are trying to find out if our model predicted the trend correctly or not.
# We can see that using plots.

real_stock_price
plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()