import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

plt.style.use("fivethirtyeight")

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



# For reading stock data from yahoo

from pandas_datareader.data import DataReader



# For time stamps

from datetime import datetime
end_date = datetime.now()

start_date = datetime(end_date.year - 1, 1, 1)
#Get the stock quote

df = DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)

#Show teh data

df
#Create a new dataframe with only the 'Close column

data = df.filter(['Close'])

data
plt.figure(figsize=(16,8))

plt.title('Close Price History')

plt.plot(data)

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price USD ($)', fontsize=18)

plt.show()
#Convert the dataframe to a numpy array

dataset = data.values

#print(dataset)

#print(int(len(dataset)*0.8))

#Get the number of rows to train the model on

training_data_len = int(np.ceil( len(dataset) * .8 ))

training_data_len
#Scale the data

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)



scaled_data
#Create the training data set

#Create the scaled training data set

train_data = scaled_data[0:int(training_data_len), :]

#Split the data into x_train and y_train data sets

x_train = []

y_train = []



for i in range(60, len(train_data)):

    x_train.append(train_data[i-60:i, 0])

    y_train.append(train_data[i, 0])

   
# Convert the x_train and y_train to numpy arrays 

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)

print(y_train.shape)
#Reshape the data

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train.shape
from keras.models import Sequential

from keras.layers import Dense, LSTM



#Build the LSTM model

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))

model.add(LSTM(50, return_sequences= False))

model.add(Dense(25))

model.add(Dense(1))
# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model

model.fit(x_train, y_train, batch_size=1, epochs=1)
#Create the testing data set

#Create a new array containing scaled values from index 1543 to 2002 

test_data = scaled_data[training_data_len - 60: , :]

#Create the data sets x_test and y_test

x_test = []

y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i, 0])
# Convert the data to a numpy array

x_test = np.array(x_test)
# Reshape the data

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
# Get the models predicted price values 

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)
# Get the root mean squared error (RMSE)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

rmse
# Plot the data

train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions

# Visualize the data

plt.figure(figsize=(16,8))

plt.title('Model')

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price USD ($)', fontsize=18)

plt.plot(train['Close'])

plt.plot(valid[['Close', 'Predictions']])

plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

plt.show()
#Show the valid and predicted prices

valid