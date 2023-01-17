#!pip install pandas

#!pip install numpy



import pandas as pd

import numpy as np

import os

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
path = "../input/international-airline-passengers.csv"

df = pd.DataFrame(pd.read_csv(path, usecols=[1]))



print("Length of the Dataframe : " + str(len(df)))

print(df.head())
dataset = df.values.astype("float32")

dataset = dataset[0:len(dataset)-1]



# Normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))



# Get the scaler details

print(scaler)



# Fit the scaler on the dataset 

dataset = scaler.fit_transform(dataset)



# Get the type of the data

print(type(dataset[0]))



# Print sample data

print(dataset)
train_size = int(len(dataset) * 0.7)

test_size = len(dataset) - train_size



# Form the training and testing datasets

train, test = dataset[0:train_size,0], dataset[train_size:len(dataset), 0]



# Get the length

print("Length of training set : " + str(len(train)))

print("Length of testing set : " + str(len(test)))



# Get the type of the train/test sets

print(type(train))

print(type(test))



print(train[0:10])

print(test[0:10])
def create_dataset_lookback(dataset, look_back=1):

    data_X = []

    data_Y = []

    for i in range(0, len(dataset)-look_back):

        x = dataset[i:(i+look_back)] # Input Data

        y = dataset[i+look_back] # Output Data

        data_X.append(x)

        data_Y.append(y)

    return np.array(data_X), np.array(data_Y)
# Training input and output

train_X, train_Y = create_dataset_lookback(train)



# Testing input and output

test_X, test_Y = create_dataset_lookback(test)



# Get the shapes

print(train_X.shape)

print(train_X[0].shape)

print(train_Y.shape)

print(train_Y[0].shape)



print(test_X.shape)

print(test_X[0].shape)

print(test_Y.shape)

print(test_Y[0].shape)



print(train_X[0:10])

print(train_Y[0:10])
# reshape input to be [samples, time steps, features]

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))



print(train_X.shape)

print(test_X.shape)
# Create and fit the LSTM network

model = Sequential()

model.add(LSTM(4, input_shape=(1, 1)))

# model.add(LSTM(8, return_sequences=True))

# model.add(LSTM(16, return_sequences=True))



# Final Model Output/Dense layer

model.add(Dense(1))



# Compile the model

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_X, train_Y, epochs=50, batch_size=1, verbose=2)
predicted_label = model.predict(test_X)

test_Y_reshape = np.reshape(test_Y, (-1,1))



# ----- Get the predicted labels before inverse transformation

print("----- Labels before inverse transformation -----")

print(predicted_label[0:10])

print(test_Y_reshape[0:10])



# Perform Inverse Transformation

predicted_label = scaler.inverse_transform(predicted_label)

test_Y_final = scaler.inverse_transform(test_Y_reshape)



# ----- Get the labels after transformation

print("----- Labels after inverse tranformation -----")

print(predicted_label[0:10])

print(test_Y_final[0:10])



# Get the shapes

print("----- Types and Shapes ----- ")

print(type(predicted_label))

print(type(test_Y_final))

print(predicted_label.shape)

print(test_Y_final.shape)



testScore = math.sqrt(mean_squared_error(test_Y_final[:,0], predicted_label[:,0]))

print("Mean squared error for the test and predicted set : " + str(testScore))
# Training input and output with lookback = 3

train_X, train_Y = create_dataset_lookback(train, 3)



# Testing input and output with lookback = 3

test_X, test_Y = create_dataset_lookback(test, 3)



# Get the shapes

print(train_X.shape)

print(train_X[0].shape)

print(train_Y.shape)

print(train_Y[0].shape)



print(test_X.shape)

print(test_X[0].shape)

print(test_Y.shape)

print(test_Y[0].shape)



print(train_X[0:10])

print(train_Y[0:10])



# Reshape input to be [samples, time steps, features]

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))



print(train_X.shape)

print(test_X.shape)



# Create and fit the LSTM network

model = Sequential()

model.add(LSTM(4, input_shape=(1, 3)))

# model.add(LSTM(8, return_sequences=True))

# model.add(LSTM(16, return_sequences=True))



# Final Model Output/Dense layer

model.add(Dense(1))



# Compile the model

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_X, train_Y, epochs=50, batch_size=1, verbose=2)
# Fit the new multivariate model

predicted_label = model.predict(test_X)

test_Y_reshape = np.reshape(test_Y, (-1,1))



# ----- Get the predicted labels before inverse transformation

print("----- Labels before inverse transformation -----")

print(predicted_label[0:10])

print(test_Y_reshape[0:10])



# Perform Inverse Transformation

predicted_label = scaler.inverse_transform(predicted_label)

test_Y_final = scaler.inverse_transform(test_Y_reshape)



# ----- Get the labels after transformation

print("----- Labels after inverse tranformation -----")

print(predicted_label[0:10])

print(test_Y_final[0:10])



# Get the shapes

print("----- Types and Shapes ----- ")

print(type(predicted_label))

print(type(test_Y_final))

print(predicted_label.shape)

print(test_Y_final.shape)



testScore = math.sqrt(mean_squared_error(test_Y_final[:,0], predicted_label[:,0]))

print("Mean squared error for the test and predicted set : " + str(testScore))