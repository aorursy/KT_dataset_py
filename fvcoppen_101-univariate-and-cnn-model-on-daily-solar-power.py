import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

import tensorflow as tf

import keras

print('tf version', tf.__version__)

print('keras version', keras.__version__)

print('numpy version', np.__version__)


solarpower = pd.read_csv("../input/solarpanelspower/PV_Elec_Gas2.csv",header = None,skiprows=1 ,names = ['date','cum_power','Elec_kW', 

                                                                            'Gas_mxm'], sep=',',usecols = [0,1,2,3],

                     

                     parse_dates={'dt' : ['date']}, infer_datetime_format=True,index_col='dt')

print(solarpower.head(2))

# make solar power stationary



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
X_valid_start_cum_power = solarpower2['2018-10-28':'2018-10-28'].cum_power.values

X_valid_start_cum_power # we need this to predict cumulative power on validation
# we devide the series into multiple input and output patterns



def my_split_window(series, window):

    '''

    the series is split in (len(series)-window)-blocks of window size, 

    y is the next value that comes after the block, 

    every block starts with the next value in the series.

    The last block ends with the last-but-one value in the series.

    '''

    X = []

    y = []

    n_steps = len(series) - window

    for step in range(n_steps):

        X.append(series[step:window+step])

        y.append(series[step + window])

    X = np.array(X)

    y = np.array(y)

    return X, y
# test my_split_window

my_series = np.array([10,20,30,40,50,60,70,80,90])

X_, y_ = my_split_window(my_series, 3)

X_, y_

# apply my_split_window on dayly solar power with a window of 365 days (we do not make account for leap years)

# the input series is the daily solar power

train_power_series = X_train.day_power.values

window = 365

X, y = my_split_window(train_power_series, window)

# print a sample

for i in range(3):

    print(X[i][-5:], y[i])


# we have an input shape = window size, number of features 

# we use only 1 feature (it is univariate) and we have a window size of one year (365 days) 

# we have to reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu', 

                                 input_shape=(window, n_features)))

model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(50, activation='relu'))

model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mae')  # metrics=['mae'])

# fit model

history = model.fit(X, y, epochs=600, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting next year

x_input = np.array(X_train.day_power[-365:]) #  next value based on data of last year

x_input = x_input.reshape((1, window, n_features)) # the model expects three dimensions as input (samples, window, features)



for i in range(365):

    y_hat = model.predict(x_input, verbose=0)

    new_x = y_hat.reshape((1,1,1))

    x_input = np.concatenate((x_input[:, -364:], new_x), axis=1)



y_predicted = x_input.reshape((x_input.shape[1]))

plt.plot(y_predicted, label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_predicted) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_predicted)

print('r2_score %.4f' % first_r2_score)

print('mae %.2f' % first_mae)
predicted_data = X_valid.copy()
predicted_data['101_predicted_from_predicted'] = y_predicted

predicted_data.to_hdf('../predicted_data.hdf5', key = 'predicted_data ', table='true', mode='a')
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

y_predicted_cumulative = cumulate(y_predicted)



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
# predicting next year

x_input = np.array(X_train.day_power[-365:]) #  next value based on data of last year

x_input = x_input.reshape((1, window, n_features))

y_hat = model.predict(x_input, verbose=0)

for i in range(365):

    new_x = y_hat.reshape((1,))

    train_power_series = np.concatenate((train_power_series, new_x), axis=0)

    X, y = my_split_window(train_power_series, window)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    history = model.fit(X, y, epochs=3, verbose=0)

    x_input = train_power_series[-365:]

    x_input = x_input.reshape((1, window, n_features))

    y_hat = model.predict(x_input, verbose=0)

    if i % 40 ==0:

        print('at i=', i, 'y_hat',y_hat)
y_predicted = x_input.reshape((x_input.shape[1]))

plt.plot(y_predicted, label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
r2_score(y_true, y_predicted) # Best possible score is 1.0 
mae = mean_absolute_error(y_true, y_predicted)

print('mae %.2f' % mae)
y_true_cumulative = cumulate(y_true)

y_predicted_cumulative = cumulate(y_predicted)



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

predicted_data['101_predicted_from_retrained'] = y_predicted

predicted_data.to_hdf('predicted_data.hdf5', key = 'predicted_data ', table='true', mode='a')