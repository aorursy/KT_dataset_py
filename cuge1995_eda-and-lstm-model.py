#load data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

price = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
calendar.head()
calendar.shape
sales.tail()
sales.shape
price.head()
price.shape
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import LSTM





def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]

	df = DataFrame(data)

	cols, names = list(), list()

	# input sequence (t-n, ... t-1)

	for i in range(n_in, 0, -1):

		cols.append(df.shift(i))

		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)

	for i in range(0, n_out):

		cols.append(df.shift(-i))

		if i == 0:

			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

		else:

			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together

	agg = concat(cols, axis=1)

	agg.columns = names

	#drop rows with NaN values

	if dropnan:

		agg.dropna(inplace=True)

	return agg





days = range(1, 1913 + 1)

time_series_columns = [f'd_{i}' for i in days]



time_series_data = sales[time_series_columns]
time_series_data.head()
# n_features = X.shape[2]

# # define model

# model = Sequential()

# model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))

# model.add(RepeatVector(n_steps_out))

# model.add(LSTM(200, activation='relu', return_sequences=True))

# model.add(TimeDistributed(Dense(n_features)))

# model.compile(optimizer='adam', loss='mse')

# # fit model

# model.fit(X, y, batch_size=16, epochs=300, verbose=0)

# # demonstrate prediction

# x_input = X[-1].reshape((1, n_steps_in, n_features))

# yhat = model.predict(x_input, verbose=0)

# print(yhat)
# # multivariate output stacked lstm example

# from numpy import array

# from numpy import hstack

# from keras.models import Sequential

# from keras.layers import LSTM

# from keras.layers import Dense



# # split a multivariate sequence into samples

# def split_sequences(sequences, n_steps):

# 	X, y = list(), list()

# 	for i in range(len(sequences)):

# 		# find the end of this pattern

# 		end_ix = i + n_steps

# 		# check if we are beyond the dataset

# 		if end_ix > len(sequences)-1:

# 			break

# 		# gather input and output parts of the pattern

# 		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]

# 		X.append(seq_x)

# 		y.append(seq_y)

# 	return array(X), array(y)



# # define input sequence

# # for i in range(100):

# #     'in_seqi%'%i = array(time_series_data.iloc[i, :])

# # in_seq2 = array(time_series_data.iloc[1, :])

# # out_seq = array(time_series_data.iloc[2, :])

# # # convert to [rows, columns] structure

# # # in_seq1 = in_seq1.reshape((len(in_seq1), 1))

# # in_seq2 = in_seq2.reshape((len(in_seq2), 1))

# # out_seq = out_seq.reshape((len(out_seq), 1))

# # horizontally stack columns

# dataset = time_series_data.iloc[:100, :]

# dataset = array(dataset)

# dataset = dataset.reshape(1913, 100)

# # choose a number of time steps

# n_steps = 28

# # convert into input/output

# X, y = split_sequences(dataset, n_steps)

# # the dataset knows the number of features, e.g. 2

# for i in range(len(X)):

# 	print(X[i], y[i])
# n_features = X.shape[2]

# # define model

# model = Sequential()

# model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

# model.add(LSTM(100, activation='relu'))

# model.add(Dense(n_features))

# model.compile(optimizer='adam', loss='mse')

# # fit model

# model.fit(X, y, epochs=400, verbose=0)

# # demonstrate prediction

# # x_input = array([[70,75,145], [80,85,165], [90,95,185]])

# # x_input = x_input.reshape((1, n_steps, n_features))

# yhat = model.predict(X[-28:].reshape((28, n_steps, n_features)), verbose=0)

# print(yhat)
# yhat = model.predict(X[0:2].reshape((2, n_steps, n_features)), verbose=0)

# print(yhat)
len(time_series_data)
import gc

times = pd.DataFrame()

def model_fit():

    model = Sequential()

    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))

    model.add(LSTM(48, return_sequences=True))

    #model.add(LSTM(128, return_sequences=True))

    #model.add(Dropout(0.3))

    model.add(LSTM(32))

    model.add(Dense(1, activation='relu'))

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])

    model.fit(train_X, train_y,batch_size=16,epochs=0,validation_data=(test_X, test_y), verbose=2, shuffle=False)

    return model



def train():

    for i in range(30490):

        a = time_series_data.iloc[i, :]

        values = np.array(a)

        # integer encode direction

        # ensure all data is float

        a = values.reshape(-1,1)

        values = a.astype('float32')

        del a

        # normalize features

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled = scaler.fit_transform(values)

        reframed = series_to_supervised(scaled, 14, 56)

        del scaled

        a, b = reframed.iloc[:, :14],reframed.iloc[:, 41]

        del reframed

        reframed1 = pd.concat([a,b],axis=1)

        values = reframed1.values

        del reframed1

        n_train_hours = 1475

        train = values[:n_train_hours, :]

        test = values[n_train_hours:1816, :]

        validation = values[1816:, :]



        # split into input and outputs

        train_X, train_y = train[:, :-1], train[:, -1]

        test_X, test_y = test[:, :-1], test[:, -1]

        vali_X, vali_y = validation[:, :-1], validation[:, -1]

        # reshape input to be 3D [samples, timesteps, features]

        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        vali_X = vali_X.reshape((vali_X.shape[0], 1, vali_X.shape[1]))

        # fit model

        model = model_fit()

        # make a prediction

        vali_hat = model.predict(vali_X)

        del model

        vali_X = vali_X.reshape((vali_X.shape[0], vali_X.shape[2]))

        # invert scaling for forecast

        vali_yhat = concatenate((vali_hat, vali_X[:, 1:]), axis=1)

        vali_yhat = scaler.inverse_transform(vali_yhat)

        vali_yhat = vali_yhat[:,0]



        times['row_%s'%i]=pd.Series(vali_yhat)

        gc.collect()

    return times



times = train()

times.to_csv('times.csv', index=False)
print('Done !!')
times.head()
times.shape
time_new = times.T

df1 = time_new.iloc[:, :28]

df2 = time_new.iloc[:, :28]

df2.columns = [f'F{i}' for i in range(1, df2.shape[1] + 1)]

df1.columns = [f'F{i}' for i in range(1, df1.shape[1] + 1)]

pre = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

ref = pd.concat([df1,df2])

ref.shape

ref = ref.reset_index()

del ref['index']



ref.insert(0, 'id', pre['id'])



num = ref._get_numeric_data()

num[num < 0] = 0

ref.to_csv('submission.csv', index=False)

ref.head()
ref.shape
# ################

# # Code from OP #

# ################

# import numpy as np

# def random_sample(len_timeseries=3000):

#     Nchoice = 600

#     x1 = np.cos(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))

#     x2 = np.cos(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))

#     x3 = np.sin(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))

#     x4 = np.sin(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))

#     y1 = np.random.random(len_timeseries)

#     y2 = np.random.random(len_timeseries)

#     y3 = np.random.random(len_timeseries)

#     for t in range(3,len_timeseries):

#         ## the output time series depend on input as follows: 

#         y1[t] = x1[t-2] 

#         y2[t] = x2[t-1]*x3[t-2]

#         y3[t] = x4[t-3]

#     y = np.array([y1,y2,y3]).T

#     X = np.array([x1,x2,x3,x4]).T

#     return y, X

# def generate_data(Nsequence = 1000):

#     X_train = []

#     y_train = []

#     for isequence in range(Nsequence):

#         y, X = random_sample()

#         X_train.append(X)

#         y_train.append(y)

#     return np.array(X_train),np.array(y_train)



# Nsequence = 100

# prop = 0.5

# Ntrain = int(Nsequence*prop)

# X, y = generate_data(Nsequence)

# X_train = X[:Ntrain,:,:]

# X_test  = X[Ntrain:,:,:]

# y_train = y[:Ntrain,:,:]

# y_test  = y[Ntrain:,:,:] 



# #X.shape = (N sequence, length of time series, N input features)

# #y.shape = (N sequence, length of time series, N targets)

# print(X.shape, y.shape)

# # (100, 3000, 4) (100, 3000, 3)



# ####################

# # Cutting function #

# ####################

# def stateful_cut(arr, batch_size, T_after_cut):

#     if len(arr.shape) != 3:

#         # N: Independent sample size,

#         # T: Time length,

#         # m: Dimension

#         print("ERROR: please format arr as a (N, T, m) array.")



#     N = arr.shape[0]

#     T = arr.shape[1]



#     # We need T_after_cut * nb_cuts = T

#     nb_cuts = int(T / T_after_cut)

#     if nb_cuts * T_after_cut != T:

#         print("ERROR: T_after_cut must divide T")



#     # We need batch_size * nb_reset = N

#     # If nb_reset = 1, we only reset after the whole epoch, so no need to reset

#     nb_reset = int(N / batch_size)

#     if nb_reset * batch_size != N:

#         print("ERROR: batch_size must divide N")



#     # Cutting (technical)

#     cut1 = np.split(arr, nb_reset, axis=0)

#     cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]

#     cut3 = [np.concatenate(x) for x in cut2]

#     cut4 = np.concatenate(cut3)

#     return(cut4)



# #############

# # Main code #

# #############

# from keras.models import Sequential

# from keras.layers import Dense, LSTM, TimeDistributed

# import matplotlib.pyplot as plt

# import matplotlib.patches as mpatches



# ##

# # Data

# ##

# N = X_train.shape[0] # size of samples

# T = X_train.shape[1] # length of each time series

# batch_size = N # number of time series considered together: batch_size | N

# T_after_cut = 100 # length of each cut part of the time series: T_after_cut | T

# dim_in = X_train.shape[2] # dimension of input time series

# dim_out = y_train.shape[2] # dimension of output time series



# inputs, outputs, inputs_test, outputs_test = \

#   [stateful_cut(arr, batch_size, T_after_cut) for arr in \

#   [X_train, y_train, X_test, y_test]]



# ##

# # Model

# ##

# nb_units = 10



# model = Sequential()

# model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),

#                return_sequences=True, units=nb_units, stateful=True))

# model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))

# model.compile(loss = 'mse', optimizer = 'rmsprop')



# ##

# # Training

# ##

# epochs = 100



# nb_reset = int(N / batch_size)

# if nb_reset > 1:

#     print("ERROR: We need to reset states when batch_size < N")



# # When nb_reset = 1, we do not need to reinitialize states

# history = model.fit(inputs, outputs, epochs = epochs, 

#                     batch_size = batch_size, shuffle=False,

#                     validation_data=(inputs_test, outputs_test))



# def plotting(history):

#     plt.plot(history.history['loss'], color = "red")

#     plt.plot(history.history['val_loss'], color = "blue")

#     red_patch = mpatches.Patch(color='red', label='Training')

#     blue_patch = mpatches.Patch(color='blue', label='Test')

#     plt.legend(handles=[red_patch, blue_patch])

#     plt.xlabel('Epochs')

#     plt.ylabel('MSE loss')

#     plt.show()



# plt.figure(figsize=(10,8))

# plotting(history) # Evolution of training/test loss



# ##

# # Visual checking for a time series

# ##

# ## Mime model which is stateless but containing stateful weights

# model_stateless = Sequential()

# model_stateless.add(LSTM(input_shape=(None, dim_in),

#                return_sequences=True, units=nb_units))

# model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))

# model_stateless.compile(loss = 'mse', optimizer = 'rmsprop')

# model_stateless.set_weights(model.get_weights())



# ## Prediction of a new set

# i = 0 # time series selected (between 0 and N-1)

# x = X_train[i]

# y = y_train[i]

# y_hat = model_stateless.predict(np.array([x]))[0]



# for dim in range(3): # dim = 0 for y1 ; dim = 1 for y2 ; dim = 2 for y3.

#     plt.figure(figsize=(10,8))

#     plt.plot(range(T), y[:,dim])

#     plt.plot(range(T), y_hat[:,dim])

#     plt.show()



# ## Conclusion: works almost perfectly.