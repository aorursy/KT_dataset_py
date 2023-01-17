import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

import tensorflow as tf

import keras

print('tf version:',tf.__version__,'\n' ,'keras version:',keras.__version__,'\n' ,'numpy version:',np.__version__)
# load previous prediction results

predicted_data = pd.read_hdf('../input/105-sol-elec-gas-2-d-multivariate-parallel-multi-o/predicted_data5.hdf5')


solarpower = pd.read_csv("../input/solarpanelspower/PV_Elec_Gas2.csv",

                         header = None,skiprows=1 ,names = ['date','cum_power','Elec_kW','Gas_mxm'], sep=',',

                         usecols = [0,1,2,3],

                     parse_dates={'dt' : ['date']}, infer_datetime_format=True,index_col='dt')

print(solarpower.head(2))

# make cum_power stationary



solarpower2 = solarpower.shift(periods=1, freq='D', axis=0)

solarpower['cum_power_shift'] = solarpower2.loc[:,'cum_power']

solarpower['day_power'] = solarpower['cum_power'].values - solarpower['cum_power_shift']

solarpower.iloc[0:1].day_power.value = 0.

A = solarpower.dropna()

del A['cum_power'], A['cum_power_shift']

solarpower = A
solarpower.head(2), solarpower.tail(2)
X_train = solarpower[:'2018-10-28']

X_valid = solarpower['2018-10-29':'2019-10-28'] # is 365 days

X_train.shape, X_valid.shape
X_train.tail(2), X_valid.head(2)
# we devide the series into multiple input and output patterns



def my_split_window(series, window_in, window_out):

    '''

    the series is split into output sequences of length window_in and 

    output sequences of lenght window_out

    returns arrays X, y

    '''

    X = []

    y = []

    n_steps = len(series) - window_in + 1

    for step in range(n_steps):

        if (step + window_in + window_out) > (len(series)):

                    break

        X_w = []

        for i in range(window_in):

            X_w.append(series[i+step])

            y_w = []

            for j in range(window_out):

                n = i + j + step + 1



                y_w.append(series[n])

        X_w = np.array(X_w)

        X.append(X_w)

        y_w = np.array(y_w)

        y.append(y_w)   

    X = np.array(X)

    y = np.array(y)

    return X, y
# test my_split_window



series = [10,20,30,40,50,60,70,80,90]

window_in = 3

window_out = 2

X_, y_ = my_split_window(series, window_in, window_out)

X_, y_

X_.shape, y_.shape
# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)



window_in = 365

window_out = 365

X, y = my_split_window(X_train.day_power.values,  window_in, window_out)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i][-2:])
X.shape
# vector output model:

# model for univariate series input and prediction of  timestep vector

# we have an input shape = (number of windows, window_in) 

#  and we have a window size of one year (365 days)

# the output vector is of shape(number of window_out)

n_features = 1 # it is a series

window_in = 365

window_out = 365

# we have to reshape from (samples, timesteps) to (samples, timesteps, n_features)

X = X.reshape((X.shape[0], X.shape[1], n_features ))



# define model

def cnn_model(window_in, window_out, n_features):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu',

                                input_shape=(window_in, n_features)))

    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dense(window_out))

    return model

    

model = cnn_model(window_in, window_out, n_features)

# compile the model:

model.compile(optimizer='adam', loss='mae')



# fit model

history = model.fit(X, y, epochs=200, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting next year based on X_valid to see if model works

# the model expects an input of shape(1, window_in, n_features  )

X_input = X_valid.day_power.ravel()

X_input = X_input.reshape(1, window_in, n_features)



y_hat = model.predict(X_input, verbose=0)
plt.plot(y_hat[0], label='predicted_power')

y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat[0]) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat[0])

print('r2_score %.5f' % first_r2_score)

print('mae %.2f' % first_mae)
# 100 epochs : 0.42520212661926315
def cumulate(series, start=0):

    '''

    start is the starting cumulative power, the series is the daily solar power

    a list with daily cumulative power is the result

    '''

    cum = [start]

    for i in range(len(series)):

        sum_plus = cum[i] + series[i]

        cum.append(sum_plus)

    return cum
y_true_cumulative = cumulate(y_true)

y_predicted_cumulative = cumulate(y_hat[0])



plt.plot(y_predicted_cumulative, label='predicted_power')

plt.plot(y_true_cumulative, label='true_power')

plt.legend()

plt.show()
true_cumulative_power_after_one_year = int(y_true_cumulative[-1])

predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])

print('true cumulative power after one year:', true_cumulative_power_after_one_year)

print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)



acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year

acc_one_year = acc_one_year * 100



print('accuracy after one year: %.2f' %  acc_one_year,'%')

print('r2 score %.2f ' % r2_score(y_true_cumulative, y_predicted_cumulative))

print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))
# adding a feature:

X_train = X_train.copy()

X_valid = X_valid.copy()

X_train['Gas_plus_Elek'] = X_train.Gas_mxm + X_train.Elec_kW

X_valid['Gas_plus_Elek'] = X_valid.Gas_mxm + X_valid.Elec_kW
# apply split window

# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)



window_in = 365

window_out = 365

X, y = my_split_window(X_train.day_power.values,  window_in, window_out)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i][-2:])
# vector output model:

# model for univariate series input and prediction of  timestep vector

# we have an input shape = (number of windows, window_in) 

#  and we have a window size of one year (365 days)

# the output vector is of shape(number of window_out)

n_features = 1 # it is a series

window_in = 365

window_out = 365

# we have to reshape from (samples, timesteps) to (samples, timesteps, n_features)

X = X.reshape((X.shape[0], X.shape[1], n_features ))

    

model = cnn_model(window_in, window_out, n_features)

# compile the model:

model.compile(optimizer='adam', loss='mae')



# fit model

history = model.fit(X, y, epochs=200, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting next year based on X_valid to see if model works

# the model expects an input of shape(1, window_in, n_features  )

X_input = X_train[-365:].day_power.ravel()

X_input = X_input.reshape(1, window_in, n_features)



y_hat = model.predict(X_input, verbose=0)
plt.plot(y_hat[0], label='predicted_power')

y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat[0]) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat[0])

print('r2_score %.5f' % first_r2_score)

print('mae %.2f' % first_mae)
y_true_cumulative = cumulate(y_true)

y_predicted_cumulative = cumulate(y_hat[0])



plt.plot(y_predicted_cumulative, label='predicted_power')

plt.plot(y_true_cumulative, label='true_power')

plt.legend()

plt.show()
true_cumulative_power_after_one_year = int(y_true_cumulative[-1])

predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])

print('true cumulative power after one year:', true_cumulative_power_after_one_year)

print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)



acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year

acc_one_year = acc_one_year * 100



print('accuracy after one year: %.2f' %  acc_one_year,'%')

print('r2 score %.5f ' % r2_score(y_true_cumulative, y_predicted_cumulative))

print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))
predicted_data['106_4f_CNN_univariate_multi_ouput_200epochs'] = y_hat[0,:]

predicted_data.to_hdf('predicted_data.hdf5',key='predicted_data', table='true',mode='a')