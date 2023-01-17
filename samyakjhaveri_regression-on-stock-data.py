import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from sklearn.preprocessing import MinMaxScaler





from keras import Sequential # To help build Deep Learning ANN/RNN

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



# import pandas_datareader as web
dataset = pd.read_csv('../input/nifty50-stock-market-data/CIPLA.csv') # Taking example of Cipla data
dataset.info()
dataset.describe()
dataset.head(5)
dataset.tail(5)
dataset['Datetime'] = pd.to_datetime(dataset['Date'])
dataset.head(5)
dataset = dataset.drop(['Date'], axis = 1) 
dataset.head(5)
dataset = dataset.set_index('Datetime')
dataset.head(5)
dataset.tail(5)
#Visualizing the closing price history for the training dataset

plt.figure(figsize=(16,8))

plt.title('Close Price History')

plt.plot(dataset['Close'])

plt.xlabel('Date',fontsize=18)

plt.ylabel('Close Price INR',fontsize=18)

plt.show()
#Creating a new dataframe with only the 'Close' column

data = dataset.filter(['Close'])

#Converting the dataframe to a numpy array

final_dataset = data.values

#Computing the number of rows to train the model on

training_data_len = math.ceil(len(final_dataset)*.75)
print("Training Dataset length:", training_data_len)
#Scale the all of the data to be values between 0 and 1 for Training Dataset

scaler = MinMaxScaler(feature_range=(0, 1)) 

scaled_data = scaler.fit_transform(final_dataset)
#Create the scaled training data set 

train_data = scaled_data[0 : training_data_len, : ]

#Split the data into x_train and y_train data sets

x_train=[]

y_train = []

for i in range(100, len(train_data)):

    x_train.append(train_data[i - 100 : i ,0])

    y_train.append(train_data[i, 0])
#Convert x_train and y_train to numpy arrays

x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data into the shape accepted by the LSTM

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#Build the LSTM network model

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

model.add(LSTM(units = 50, return_sequences = False))

model.add(Dense(units = 25))

model.add(Dense(units = 1))
#Compile the model

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Train the model

model.fit(x_train, y_train, batch_size = 1, epochs = 1)
#Test data set

test_data = scaled_data[training_data_len - 100: , : ]
print(len(test_data))
#Create the x_test and y_test data sets

x_test = []

y_test =  final_dataset[training_data_len:,:] 

for i in range(100, len(test_data)):

    x_test.append(test_data[i - 100 : i, 0])
#Convert x_test to a numpy array 

x_test = np.array(x_test)
#Reshape the data into the shape accepted by the LSTM

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Getting the models predicted price values

predictions = model.predict(x_test) 

predictions = scaler.inverse_transform(predictions)#Undo scaling
#Calculate/Get the value of RMSE

rmse = np.sqrt(np.mean(((predictions - y_test)**2)))

rmse
#Plot/Create the data for the graph

train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions

#Visualize the data

plt.figure(figsize=(16,8))

plt.title('Model')

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price INR', fontsize=18)

plt.plot(train['Close'])

plt.plot(valid[['Close', 'Predictions']])

plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

plt.show()
valid
#Get the quote

cipla_quote = dataset.tail(100)
cipla_quote.head(5)
#Create a new dataframe

new_dataset = cipla_quote.filter(['Close'])
#Get the last 100 day closing price 

last_100_days = new_dataset[-100:].values
#Scale the data to be values between 0 and 1

last_100_days_scaled = scaler.transform(last_100_days)
#Create an empty list

X_test = []

#Append teh past 100 days

X_test.append(last_100_days_scaled)
#Convert the X_test data set to a numpy array

X_test = np.array(X_test)

#Reshape the data

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price

pred_price = model.predict(X_test)

#undo the scaling 

pred_price = scaler.inverse_transform(pred_price)

print(pred_price)
#Get the quote

cipla_quote2 = dataset.tail(1)

print(cipla_quote2['Close'])