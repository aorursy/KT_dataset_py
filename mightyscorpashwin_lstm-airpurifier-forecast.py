# This code is using LSTM for a timeseries forecasting example. I am trying to workout the LSTM examples explained on the website https://machinelearningmastery.com by Jason brownnie.
from pandas import DataFrame

from pandas import Series

from pandas import concat

from pandas import read_csv

from pandas import datetime

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from math import sqrt

from matplotlib import pyplot

import numpy



import os

print(os.listdir("../input/airpurifiersales"))
# date parser

def parser_dt (x):

    return datetime.strptime('190'+x, '%Y-%m')
def difference(dataset,interval=1):

    diff=list()

    for i in range(interval,len(dataset)):

        value= dataset[i] - dataset[i-interval]

        diff.append(value)

    return Series(diff)
# frame a sequence as a supervised learning problem

def timeseries_to_supervised(data, lag=1):

	df = DataFrame(data)

	columns = [df.shift(i) for i in range(1, lag+1)]

	columns.append(df)

	df = concat(columns, axis=1)

	df.fillna(0, inplace=True)

	return df
# scale train and test data to [-1, 1]

def scale(train, test):

	# fit scaler

	scaler = MinMaxScaler(feature_range=(-1, 1))

	scaler = scaler.fit(train)

	# transform train

	train = train.reshape(train.shape[0], train.shape[1])

	train_scaled = scaler.transform(train)

	# transform test

	test = test.reshape(test.shape[0], test.shape[1])

	test_scaled = scaler.transform(test)

	return scaler, train_scaled, test_scaled
# fit an LSTM network to training data

def fit_lstm(train, batch_size, nb_epoch, neurons):

	X, y = train[:, 0:-1], train[:, -1]

	X = X.reshape(X.shape[0], 1, X.shape[1])

	model = Sequential()

	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))

	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')

	for i in range(nb_epoch):

		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)

		model.reset_states()

	return model
# inverse scaling for a forecasted value

def invert_scale(scaler, X, value):

	new_row = [x for x in X] + [value]

	array = numpy.array(new_row)

	array = array.reshape(1, len(array))

	inverted = scaler.inverse_transform(array)

	return inverted[0, -1]
# make a one-step forecast

def forecast_lstm(model, batch_size, X):

	X = X.reshape(1, 1, len(X))

	yhat = model.predict(X, batch_size=batch_size)

	return yhat[0,0]
# invert differenced value

def inverse_difference(history, yhat, interval=1):

	return yhat + history[-interval]
data=read_csv('../input/airpurifiersales/airpurifiersales.csv',header=0)
print(data.shape)

print(data.columns)
# transform data to stationary

row_values = data.values

diff_values=difference(row_values,1)
# print(diff_values)





# 0        [1]

# 1       [-1]

# 2        [3]

# 3       [-2]

# 4        [1]

# 5        [1]
# transform data to be supervised learning

supervised = timeseries_to_supervised(diff_values, 1)

supervised_values = supervised.values





#          0      0

# 0        0    [1]

# 1      [1]   [-1]

# 2     [-1]    [3]

# 3      [3]   [-2]

# 4     [-2]    [1]

# 5      [1]    [1]

# 6      [1]   [-1]

# 7     [-1]   [-1]

# 8     [-1]    [0]

# 9      [0]   [-1]

# 10    [-1]    [0]
print(supervised_values.shape)
# split data into train and test-sets

train, test = supervised_values[0:-88], supervised_values[-88:]

print(test[0:10])

print(train[0:10])
print(train.shape)

print(test.shape)



# (355, 2)

# (88, 2)
# transform the scale of the data

scaler, train_scaled, test_scaled = scale(train, test)
# print(train_scaled[0:10])



# [[0.13227513 0.14285714]

#  [0.14285714 0.12169312]

#  [0.12169312 0.16402116]

#  [0.16402116 0.11111111]

#  [0.11111111 0.14285714]

#  [0.14285714 0.14285714]

#  [0.14285714 0.12169312]

#  [0.12169312 0.12169312]

#  [0.12169312 0.13227513]

#  [0.13227513 0.12169312]]



# print(test_scaled[0:10])

# [[ 0.18518519  0.3015873 ]

#  [ 0.3015873  -0.00529101]

#  [-0.00529101  0.12169312]

#  [ 0.12169312  0.20634921]

#  [ 0.20634921  0.12169312]

#  [ 0.12169312  0.07936508]

#  [ 0.07936508  0.16402116]

#  [ 0.16402116  0.16402116]

#  [ 0.16402116  0.05820106]

#  [ 0.05820106  0.15343915]]
# fit the model

lstm_model = fit_lstm(train_scaled, 1, 3500, 4)

# forecast the entire training dataset to build up state for forecasting

train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)

lstm_model.predict(train_reshaped, batch_size=1)
# walk-forward validation on the test data

predictions = list()

for i in range(len(test_scaled)):

	# make one-step forecast

	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]

	yhat = forecast_lstm(lstm_model, 1, X)

	# invert scaling

	yhat = invert_scale(scaler, X, yhat)

	# invert differencing

	yhat = inverse_difference(row_values, yhat, len(test_scaled)+1-i)

	# store forecast

	predictions.append(yhat)

	expected = row_values[len(train) + i + 1]

	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

 

# report performance

rmse = sqrt(mean_squared_error(row_values[-88:], predictions))

print('Test RMSE: %.3f' % rmse)

# line plot of observed vs predicted

pyplot.figure(figsize=(15,10))

pyplot.plot(row_values[-88:])

pyplot.plot(predictions)

pyplot.show()
# serialize model to JSON

model_json = lstm_model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

lstm_model.save_weights("model.h5")

print("Saved model to disk")