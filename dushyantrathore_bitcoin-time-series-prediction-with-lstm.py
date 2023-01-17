from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

import pandas as pd

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

import plotly.offline as py

import plotly.graph_objs as go

import numpy as np

import seaborn as sns

py.init_notebook_mode(connected=True)

%matplotlib inline
data = pd.read_csv(filepath_or_buffer="../input/btcusdkraken/BTCUSDKRAKEN", index_col="Date")



# Get the number of columns of the dataframe

print("Columns : " + str(data.columns.values))

# Get the shape of the dataframe

print("Shape : " + str(data.shape))

# Get the head(), here the first 5 elements of the dataframe

print(data.head(5))
data['Weighted Price'].replace(0, np.nan, inplace=True)

data['Weighted Price'].fillna(method='ffill', inplace=True)



# Get the head(), here the first 5 elements of the dataframe

print(data.head(5))
from sklearn.preprocessing import MinMaxScaler

values = data['Weighted Price'].values.reshape(-1,1)

print(values[0])

values = values.astype('float32')

print(values[0])

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



# Get the type of the new item scaled

print(type(scaled))



# Get the length of the new item scaled

print("Length of the new datframe : " + str(len(scaled)))



# Get the first 5 elements from the scaled dataframe

print(scaled[0:5,])
train_size = int(len(scaled) * 0.7)

print("Train Size : " + str(train_size))

test_size = len(scaled) - train_size

print("Test Size : " + str(test_size))

# print(scaled[0,])

train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]

print("Length of training data : " + str(len(train)))

print("Length of testing data : " + str(len(test)))
def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):

        a = dataset[i:(i + look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)
look_back = 1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)



print(trainX.shape)

print(trainY.shape)

print(testX.shape)

print(testY.shape)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



print(trainX.shape)

print(testX.shape)



# print(trainX)

# print(trainY)
# Initialise the sequential model

model = Sequential()

# Add the LSTM hidden layer with 100 units

model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))

# Add the output layer

model.add(Dense(1))

# Compile the model with Mean Absolute Error as the loss factor and ADAM as the optimiser

model.compile(loss='mae', optimizer='adam')

# Fit the model using the training and testing data

history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=1, shuffle=False)
pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()
yhat = model.predict(testX) # Here yhat is the predicted value from the test set (y_pred)

print(yhat.shape)

print(yhat[0])



pyplot.plot(yhat, label='predict')

pyplot.plot(testY, label='true')

pyplot.legend()

pyplot.show()
# scaler = MinMaxScaler(feature_range=(0, 1)) as used before for fit_transform and MinMaxScaler

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))

testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))



print(yhat_inverse.shape)

print(testY_inverse.shape)



print(yhat_inverse[0])

print(testY_inverse[0])
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))

print('Test RMSE: %.3f' % rmse)
pyplot.plot(yhat_inverse, label='predict')

pyplot.plot(testY_inverse, label='actual', alpha=0.5)

pyplot.legend()

pyplot.show()