import pandas as pd
import numpy as np
df = pd.read_csv('../input/h-p-c/household_power_consumption.txt', sep = ';', header=0, low_memory=False, infer_datetime_format=True,parse_dates={'datetime':[0,1]}, index_col=['datetime'])

df.replace('?', np.nan, inplace=True)
df = df.astype('float32')
df
df.isna().sum()
def fill_missing(values):
    one_day = 60*24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if np.isnan(values[row, col]):
                values[row,col] = values[row-one_day,col]

fill_missing(df.values)
values = df.values
df['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
daily_groups = df.resample('D')
daily_data = daily_groups.sum()

sub_metering_remainder = (df['Global_active_power'] * 1000 / 60) -(df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])

df.to_csv('household_power_consumption.csv')
def evaluate_forecasts(actual, predicted):
    scores = list()
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:,i], predicted[:,i])
        rmse = sqrt(mse)
        scores.append(rmse)
    s=0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row,col])**2
    score = sqrt(s/(actual.shape[0]*actual.shape[1]))
    return score, scores
from numpy import split
from numpy import array
from pandas import read_csv
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test


train, test = split_dataset(daily_data.values)
# evaluate a single model
def evaluate_model(train, test, n_input):
	model = build_model(train, n_input)
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		yhat_sequence = forecast(model, history, n_input)
		predictions.append(yhat_sequence)
		history.append(test[i, :])
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0,70, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

from numpy import array
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM


n_input = 7
score_lstm, scores_lstm = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score_lstm, scores_lstm)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores_lstm, marker='o', label='lstm')
# pyplot.plot(days, scores_1d, marker='x', label='lstm')


pyplot.show()
def build_model_dec_1d(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
n_input = 14
score_1d, scores_1d = evaluate_model(train, test, n_input)
# summarize scores

summarize_scores('lstm', score_1d, scores_1d)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores_1d, marker='x', label='lstm')
# pyplot.plot(days, scores_lstm, marker='x', label='lstm')


pyplot.show()

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
n_input = 14
score_conv, scores_conv = evaluate_model(train, test, n_input)

# summarize scores
summarize_scores('lstm', score_conv, scores_conv)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores_conv, marker='*', label='lstm')
pyplot.show()
# train the model
def build_model(train, n_steps, n_length, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape into subsequences [samples, time steps, rows, cols, channels]
	train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
def evaluate_model(train, test, n_steps, n_length, n_input):
	# fit model
	model = build_model(train, n_steps, n_length, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
# make a forecast
def forecast(model, history, n_steps, n_length, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [samples, time steps, rows, cols, channels]
	input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
n_steps, n_length = 2, 7
# define the total days to use as input
n_input = n_length * n_steps
score_cl, scores_cl = evaluate_model(train, test, n_steps, n_length, n_input)

# summarize scores
summarize_scores('lstm', score_cl, scores_cl)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores_cl, marker='+', label='lstm')
pyplot.show()
pyplot.plot(days, scores_cl, marker='+', label='lstm')
pyplot.plot(days, scores_conv, marker='*', label='lstm')
pyplot.plot(days, scores_lstm, marker='o', label='lstm')
pyplot.plot(days, scores_1d, marker='x', label='lstm')
pyplot.legend(['ConvLSTM', 'Conv', 'LSTM', 'Enc-Dec'], loc='upper left')
pyplot.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.layers import Dropout, Dense, SimpleRNN
from keras.datasets import mnist
from keras.models import Sequential
from keras. layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D

df = pd.read_csv('../input/apple-stocks/AAPL.csv', sep=',', parse_dates=['Date'])
df.head()
def convertToMatrix(data, step):
  X, Y =[], []
  for i in range(len(data)-step):
    d=i+step
    X.append(data[i:d,])
    Y.append(data[d,])
  return np.array(X), np.array(Y)
values=df['Close'].values
train3, test3 = values[0:200], values[200:253]
s = 7

test3 = np.append(test3,np.repeat(test3[-1,],s))
train3 = np.append(train3,np.repeat(train3[-1,],s))
trainX3,trainY3 =convertToMatrix(train3,s)
testX3,testY3 =convertToMatrix(test3,s)
trainX3 = np.reshape(trainX3, (trainX3.shape[0], 1, trainX3.shape[1]))
testX3 = np.reshape(testX3, (testX3.shape[0], 1, testX3.shape[1]))
model3 = Sequential()
model3.add(LSTM(200, activation = 'relu', input_shape = (1, s)))
model3.add(Dense(1))
model3.compile(loss = 'mse', optimizer = 'adam')
l1 = model3.fit(trainX3, trainY3, epochs = 100, batch_size=25)
# Ошибка на тестовой выборке
score3 = model3.evaluate(testX3, testY3)
plt.figure(figsize = (15,6))

plt.plot(testY3)
plt.plot(model3.predict(testX3))
plt.title('Сравнение реальных и предсказанных значений по модели LSTM')
plt.legend(['real', 'predicted'], loc='upper left')

model4 = Sequential()
model4.add(LSTM(200, activation='relu', input_shape=(1, s)))
model4.add(RepeatVector(1))
model4.add(LSTM(200, activation='relu', return_sequences=True))
model4.add(TimeDistributed(Dense(50, activation='relu')))
model4.add(TimeDistributed(Dense(1)))
model4.compile(loss='mse', optimizer='adam')
l2 = model4.fit(trainX3, trainY3, epochs = 100, batch_size=25)
# Ошибка на тестовой выборке
score4 = model4.evaluate(testX3, testY3)

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
n_steps = 7
n_length = 1
train_x5 = trainX3.reshape((trainX3.shape[0], n_steps, 1, n_length, 1))
train_y5 = trainY3.reshape((trainY3.shape[0], 1, 1))

model5 = Sequential()
model5.add(ConvLSTM2D(filters=64, kernel_size=(1), activation='relu', input_shape = (n_steps, 1, n_length, 1)))
model5.add(Flatten())
model5.add(RepeatVector(1))
model5.add(LSTM(200, activation='relu', return_sequences=True))
model5.add(Dense(100, activation='relu'))
model5.add(Dense(1))
model5.compile(loss='mse', optimizer='adam')
l3 = model5.fit(train_x5, train_y5, epochs = 100, batch_size=25)


model6 = Sequential()
model6.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, s)))
model6.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model6.add(MaxPooling1D(pool_size=1))
model6.add(Flatten())
model6.add(Dense(100, activation='relu'))
model6.add(Dense(1))
model6.compile(loss='mse', optimizer='adam')
l4 = model6.fit(trainX3, trainY3, epochs = 100, batch_size=25)
plt.figure(figsize = (15,6))

plt.plot(testY3)
plt.plot(model6.predict(testX3))
plt.title('Сравнение реальных и предсказанных значений по модели Кодировщик Conv')
plt.legend(['real', 'predicted'], loc='upper left')

plt.figure(figsize = (15,6))
plt.plot(l1.history['loss'])
plt.plot(l2.history['loss'])
plt.plot(l3.history['loss'])
plt.plot(l4.history['loss'])
plt.yscale('log')
plt.legend(['lstm', 'enc-dec', 'convLSTM2D', 'кодировщик conv'], loc='upper left')


