import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import sklearn
bits = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

bits.head()
bits['date'] = pd.to_datetime(bits['Timestamp'],unit='s').dt.date

group = bits.groupby('date')

Real_Price = group['Weighted_Price'].mean()
Real_Price[0:5]
# split data

prediction_days = 30

df_train= Real_Price[:len(Real_Price)-prediction_days]

df_test= Real_Price[len(Real_Price)-prediction_days:]
# hyperparameters

n_steps = 10

nb_epochs=20

lr=0.001
# Data preprocess

training_set = df_train.values

training_set = np.reshape(training_set, (len(training_set), 1))

test_set = df_test.values

test_set = np.reshape(test_set, (len(test_set), 1))

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

training_set = sc.fit_transform(training_set)

test_set = sc.transform(test_set)
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

	X, y = list(), list()

	for i in range(len(sequence)):

		# find the end of this pattern

		end_ix = i + n_steps

		# check if we are beyond the sequence

		if end_ix > len(sequence)-1:

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

		X.append(seq_x)

		y.append(seq_y)

	return np.array(X), np.array(y)
#Prepare batches bs*seq_len*1

X_train,y_train = split_sequence(training_set, n_steps)

print(X_train.shape)

print(len(test_set))

X_test, y_test = split_sequence(test_set, n_steps)

print(X_test.shape)
import tensorflow as tf
mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# define model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1)))

model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[mc, es])
# load the saved model

saved_model = tf.keras.models.load_model('best_model.h5')

# evaluate the model

train_loss = saved_model.evaluate(X_train, y_train, verbose=0)

test_loss = saved_model.evaluate(X_test, y_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_loss, test_loss))
train_predictions = saved_model.predict(X_train)

test_predictions = saved_model.predict(X_test)
train_p = sc.inverse_transform(train_predictions)

test_p = sc.inverse_transform(test_predictions)
import math

t = np.linspace(0, 2*math.pi, 1425)

plt.plot(t, sc.inverse_transform(y_train),'r')

plt.plot(t,train_p,'g')

plt.show()
t = np.linspace(0, 2*math.pi, 20)

plt.plot(t, sc.inverse_transform(y_test),'r')

plt.plot(t,test_p,'g')

plt.show()
# define model

bi_model = tf.keras.models.Sequential()

bi_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='relu'), input_shape=(n_steps, 1)))

bi_model.add(tf.keras.layers.Dense(1))

bi_model.compile(optimizer='adam', loss='mse')
bi_model.fit(X_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[mc, es])
# load the saved model

saved_model2 = tf.keras.models.load_model('best_model.h5')

# evaluate the model

train_loss = saved_model2.evaluate(X_train, y_train, verbose=0)

test_loss = saved_model2.evaluate(X_test, y_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_loss, test_loss))
train_predictions = saved_model2.predict(X_train)

test_predictions = saved_model2.predict(X_test)
train_p = sc.inverse_transform(train_predictions)

test_p = sc.inverse_transform(test_predictions)
import math

t = np.linspace(0, 2*math.pi, 1425)

plt.plot(t, sc.inverse_transform(y_train),'r')

plt.plot(t,train_p,'g')

plt.show()
t = np.linspace(0, 2*math.pi, 20)

plt.plot(t, sc.inverse_transform(y_test),'r', label="True label")

plt.plot(t,test_p,'g', label="Predictions")

plt.legend()

plt.show()
def split_sequence(sequence, n_steps_in, n_steps_out):

	X, y = list(), list()

	for i in range(len(sequence)):

		# find the end of this pattern

		end_ix = i + n_steps_in

		out_end_ix = end_ix + n_steps_out

		# check if we are beyond the sequence

		if out_end_ix > len(sequence):

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

		X.append(seq_x)

		y.append(seq_y)

	return np.array(X), np.array(y)
# hyperparameters

n_steps_out = 3
#Prepare batches bs*seq_len*1

X_train,y_train = split_sequence(training_set, n_steps, n_steps_out)

print(X_train.shape, y_train.shape)

X_test, y_test = split_sequence(test_set, n_steps, n_steps_out)

print(X_test.shape, y_test.shape)
# define model

ed_model = tf.keras.models.Sequential()

ed_model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(n_steps, 1)))

ed_model.add(tf.keras.layers.RepeatVector(n_steps_out))

ed_model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))

ed_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

ed_model.compile(optimizer='adam', loss='mse')
nb_epochs = 50
history = ed_model.fit(X_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[mc, es])
# plot training history

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.title("Training Losses")

plt.show()
test_predictions = ed_model.predict(X_test)

test_predictions.shape
y_pred = test_predictions[:,2,:]
y_true = y_test[:,2,:]

y_true.shape
y_pred.shape
t = np.linspace(0, 2*math.pi, 18)

plt.plot(t, sc.inverse_transform(y_true),'r', label="True Data")

plt.plot(t,sc.inverse_transform(y_pred),'g', label = "Prediction")

plt.legend()

plt.show()
ed_model.summary()