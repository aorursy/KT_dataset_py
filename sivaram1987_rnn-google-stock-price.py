# Recurrent Neural Network 



#Data Preprocessing 

 

# Importing the libraries 

import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd 

import os
# Importing the training set 

dataset_train = pd.read_csv("../input/google-stock-price/Google_Stock_Price_Train.csv") 

training_set = dataset_train.iloc[:, 1:2].values 

dataset_train.head()
# Feature Scaling 

from sklearn.preprocessing import MinMaxScaler 

sc = MinMaxScaler(feature_range = (0, 1)) 

training_set_scaled = sc.fit_transform(training_set) 

training_set_scaled 
# Creating a data structure with 60 timesteps and 1 output 

# ie v r training on the last 60 days 

# and predicting the next timestamp/days value 

# When v tried with 1 ts the model was overfitting 

# With 30,40,50 timestamps v did not get a good model 

# as it wsa not capturing the model 

# the best was 60 financial days 

# ie 3 months as ea month has 20 financial days 

X_train = [] #60 prev stock prices before the financial day 

    # this is the ip to the RNN 

y_train = []  # will contain the stock price the next fin day 

    # this is the op  
# since v need 60 prev days to start predicting frm 61st day v r starting at 60 ie 61 

for i in range(60, 1258): # upper bound is last row, lower bound is i-60 

    X_train.append(training_set_scaled[i-60:i, 0]) 

    y_train.append(training_set_scaled[i, 0]) 

X_train, y_train = np.array(X_train), np.array(y_train) 

print(X_train) 

print("#********************") 

print(y_train) 
print(X_train.shape[0]) 
print(X_train.shape[1]) 
# Reshaping 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 

print(X_train) 
# Building the RNN 

 

# Importing the Keras libraries and packages 

from keras.models import Sequential 

from keras.layers import Dense 

from keras.layers import LSTM 

from keras.layers import Dropout
# Initialising the RNN 

regressor = Sequential() 
# Adding the first LSTM layer and some Dropout regularisation 

# for preventing overfitting 

# this is the first lstm layer 

# v want v high dimensionality 

# v will start with 50 but v can increase it as much as v desire 

# capturing trend in stock time series is v complex 

# v can also choose 3 to 5 neruons but it will not b able to  

# capture the trend 

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) 

# 50 is the num of neurons or cells 

# True for stacked lstm, v r adding another layer of lstm 

# on top of it 
# inp_shape 

regressor.add(Dropout(0.2)) 

# 20% of 50 is 10, 10 neurons will b droppd  

# ie 20% of ns will b ignord dur training 

# ie dur frward & bw prop 

# during ea iteration 
# Adding a second LSTM layer and some Dropout regularisation 

#regressor.add(LSTM(units = 50, return_sequences = True)) 

#regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation 

#regressor.add(LSTM(units = 50, return_sequences = True)) 

#regressor.add(Dropout(0.2)) 
# Adding a fourth LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50)) 

regressor.add(Dropout(0.2)) 
# Adding the output layer 

regressor.add(Dense(units = 1)) 

# 1 is the dimension of the op layer 

# ie 1 neruon 
# Compiling the RNN 

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 
# Fitting the RNN to the Training set 

#regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

regressor.fit(X_train, y_train, epochs = 10, batch_size = 32) 

# every 32 stock prices back prop is going to happn 
regressor = Sequential() 
# Adding the first LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) 

regressor.add(Dropout(0.2)) 
# Adding a second LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50, return_sequences = True)) 

regressor.add(Dropout(0.2)) 
# Adding a third LSTM layer and some Dropout regularisation 

#regressor.add(LSTM(units = 50, return_sequences = True)) 

#regressor.add(Dropout(0.2)) 
# Adding a fourth LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50)) 

regressor.add(Dropout(0.2)) 
# Adding the output layer 

regressor.add(Dense(units = 1)) 
# Compiling the RNN 

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 
# Fitting the RNN to the Training set 

#regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

regressor.fit(X_train, y_train, epochs = 10, batch_size = 32) 
regressor = Sequential() 

 

# Adding the first LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) 

regressor.add(Dropout(0.2)) 
# Adding a second LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50, return_sequences = True)) 

regressor.add(Dropout(0.2)) 
# Adding a third LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50, return_sequences = True)) 

regressor.add(Dropout(0.2)) 
# Adding a fourth LSTM layer and some Dropout regularisation 

regressor.add(LSTM(units = 50)) 

regressor.add(Dropout(0.2)) 
# Adding the output layer 

regressor.add(Dense(units = 1)) 
# Compiling the RNN 

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 
# Fitting the RNN to the Training set 

#regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

regressor.fit(X_train, y_train, epochs = 10, batch_size = 32) 
# Making the predictions and visualising the results 

 

# Getting the real stock price of 2017 

dataset_test = pd.read_csv("../input/google-stock-pricetest/Google_Stock_Price_Test.csv") 

real_stock_price = dataset_test.iloc[:, 1:2].values 
#Getting the predicted stock price of 2017 

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) 

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values 

inputs = inputs.reshape(-1,1) 

inputs = sc.transform(inputs) 

X_test = [] 

for i in range(60, 80): 

    X_test.append(inputs[i-60:i, 0]) 

X_test = np.array(X_test) 

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

predicted_stock_price = regressor.predict(X_test) 

predicted_stock_price = sc.inverse_transform(predicted_stock_price) 
# Visualising the results 

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price') 

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price') 

plt.title('Google Stock Price Prediction') 

plt.xlabel('Time') 

plt.ylabel('Google Stock Price') 

plt.legend() 

plt.show() 