import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

import tensorflow as tf

import keras

print('tf version:',tf.__version__,'\n' ,'keras version:',keras.__version__,'\n' ,'numpy version:',np.__version__)
# read the data file

solarpower = pd.read_csv("../input/solarpanelspower/PV_Elec_Gas2.csv",header = None,skiprows=1 ,

                    names = ['date','cum_power','Elec_kW', 'Gas_mxm'], sep=',',usecols = [0,1,2,3],

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
# funcion to reverse make stationary

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
solarpower.head(2), solarpower.tail(2)
# add a feature

solarpower['Gas_plus_Elek'] = solarpower.Gas_mxm + solarpower.Elec_kW

solarpower['Gas_plus_Elek'] = solarpower.Gas_mxm + solarpower.Elec_kW
X_train = solarpower[:'2018-10-28']

X_valid = solarpower['2018-10-29':'2019-10-28'] # is 365 days

X_train.shape, X_valid.shape
X_train.tail(2), X_valid.head(2)
X_valid_start_cum_power = solarpower2['2018-10-28':'2018-10-28'].cum_power.values

X_valid_start_cum_power # we need this to predict cumulative power on validation
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
X = np.load('../input/107e-windowed-data/107E_X_train_windowed_1year.npy')

y = np.load('../input/107e-windowed-data/107E_y_train_windowed_1year.npy')
X_train.columns
# vector output model:

# model for univariate series input and prediction of  timestep vector

# we have an input shape = (number of windows, window_in) 

#  and we have a window size of one year (365 days)

# the output vector is of shape(number of window_out)

window_in = 365

window_out = 365

n_features = X.shape[2]

print('n_features',n_features)

features = ['day_power', 'Elec_kW', 'Gas_mxm', 'Gas_plus_Elek']

# define model

def multi_step_output_model(window_in, window_out, n_features):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', 

                                 input_shape=(window_in, n_features)))

    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.BatchNormalization())    

    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(window_out))

    return model



model = multi_step_output_model(window_in, window_out, window_out)

epochs=10000

range1 = 4

y_hat_dict = {}

for steps in range(range1):

    # compile the model:

    model.compile(optimizer='adam', loss='mae')

    # fit model

    history = model.fit(X, y, epochs=epochs, verbose=0)

    X_input = np.array(X_train[features][-365:].values)

    X_input = X_input.reshape(1, window_in, n_features)

    y_hat = model.predict(X_input, verbose=0)

    name = '107E_y_hat_10000e' + str(steps)

    y_hat_dict[name]=y_hat[0]

    file = name + 'range' + str(range1) + '.npy'

    np.save(file, y_hat[0])

    print('step', steps, 'done')
y_true = X_valid.day_power.values



plt.plot(y_true, label='true_power')

for key, value in y_hat_dict.items()  :

    plt.plot(value, label=key)

    plt.legend()

    first_r2_score = r2_score(y_true, value) # Best possible score is 1.0 

    first_mae = mean_absolute_error(y_true, value)

    print('r2_score %.5f' % first_r2_score)

    print('mae %.2f' % first_mae)

plt.legend()

plt.show()



y_true_cumulative = cumulate(y_true)

plt.plot(y_true_cumulative, label='true_power')

for key, value in y_hat_dict.items()  :

    y_predicted_cumulative = cumulate(value)

    plt.plot(y_predicted_cumulative, label='predicted_power')

    true_cumulative_power_after_one_year = int(y_true_cumulative[-1])

    predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])

    print('true cumulative power after one year:', true_cumulative_power_after_one_year)

    print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)

    acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year

    acc_one_year = acc_one_year * 100

    print('accuracy after one year: %.2f' %  acc_one_year,'%')

    print('r2 score %.2f ' % r2_score(y_true_cumulative, y_predicted_cumulative))

    print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))

    

plt.legend()

plt.show()