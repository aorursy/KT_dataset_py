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
X_train = solarpower[:'2018-10-28']

X_valid = solarpower['2018-10-29':'2019-10-28'] # is 365 days

X_train.shape, X_valid.shape
# we devide the series into multiple input and output patterns



def my_split_window(array, y_series, window_in, window_out):

    '''

    the Pandas dataframe is split into output sequences of length window_in and 

    output sequences of lenght window_out

    returns arrays X, y

    '''

    X = []

    y = []

    n_steps = array.shape[0] - window_in + 1

    #print('n_steps', n_steps)

    for step in range(n_steps):

        if (step + window_in + window_out -1) > (len(y_series)):

            break

        X_w = []

        for i in range(window_in):

            X_w.append(array[i+step])

            y_w = []

            for j in range(window_out):

                n = i + j + step

                y_w.append(y_series[n])

        X_w = np.array(X_w)

        X.append(X_w)

        y_w = np.array(y_w)

        y.append(y_w)   

    X = np.array(X)

    y = np.array(y)

    return X, y
# adding a feature with simple feature engineering



X_train = X_train.copy()

X_valid = X_valid.copy()

X_train['Gas_plus_Elek'] = X_train.Gas_mxm + X_train.Elec_kW

X_valid['Gas_plus_Elek'] = X_valid.Gas_mxm + X_valid.Elec_kW
# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)

# the input series is the daily solar power



window_in = 365

window_out = 365

features = ['Elec_kW', 'Gas_mxm', 'Gas_plus_Elek']

y_series = X_train.day_power.values

X, y = my_split_window(np.array(X_train[features]) , y_series ,  window_in, window_out)

print('X.shape', X.shape, 'y.shape', y.shape)

# print a sample

for i in range(2):

    print(X[i][-2:], y[i][-2:])
print('X.shape', X.shape)
# vector output model:

# model for univariate series input and prediction of  timestep vector

# we have an input shape = (number of windows, window_in) 

#  and we have a window size of one year (365 days)

# the output vector is of shape(number of window_out)



n_features = 3

# define model



def cnn_model(window_in, n_features, filters=10):

    visible = tf.keras.layers.Input(shape=(window_in, n_features))

    cnn = tf.keras.layers.Conv1D(filters=filters, kernel_size=2, activation='relu')(visible)

    cnn = tf.keras.layers.MaxPool1D(pool_size=2)(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)

    cnn = tf.keras.layers.Dense(window_out, activation='relu')(cnn)

    cnn = tf.keras.layers.Dropout(0.2)(cnn)

    cnn = tf.keras.layers.Conv1D(filters=filters ,kernel_size=2, activation='relu')(cnn)

    cnn = tf.keras.layers.MaxPool1D(pool_size=2)(cnn)

    cnn = tf.keras.layers.Flatten()(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)

    cnn = tf.keras.layers.Dense(window_out, activation='relu')(cnn)

    cnn = tf.keras.layers.Dropout(0.2)(cnn)

    return visible, cnn



visible1, cnn1 = cnn_model(window_in, n_features, filters=1)

visible2, cnn2 = cnn_model(window_in, n_features, filters=2)

visible3, cnn3 = cnn_model(window_in, n_features, filters=3)

visible4, cnn4 = cnn_model(window_in, n_features, filters=4)

visible5, cnn5 = cnn_model(window_in, n_features, filters=5)

visible6, cnn6 = cnn_model(window_in, n_features, filters=6)





merge = tf.keras.layers.concatenate([cnn1, cnn2, cnn3, cnn4, cnn5, cnn6])

dense = tf.keras.layers.Dense(window_out, activation='relu')(merge)

output = tf.keras.layers.Dense(window_out)(merge)

model = tf.keras.Model(inputs=[visible1,visible2, visible3, visible4,visible5, visible6], outputs = output)



# compile the model:

model.compile(optimizer='adam', loss='mae')



# fit model

history = model.fit([X, X , X, X, X, X], y, epochs=10, verbose=1)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting next year on X_train last year 

# the model expects an input of shape(1, window_in, n_features  )

X_input = np.array(X_train[features][-365:].values)

X_input = X_input.reshape(1, window_in, n_features)



y_hat = model.predict([X_input, X_input, X_input, X_input, X_input, X_input], verbose=0)
# plot predicted an true values

plt.plot(y_hat[0], label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
# print relevant scores

first_r2_score = r2_score(y_true, y_hat[0]) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat[0])

print('r2_score %.5f' % first_r2_score)

print('mae %.2f' % first_mae)
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
# train more



# fit model

history = model.fit([X,X,X,X,X,X], y, epochs=10000, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()





# predicting next year on X_train last year 

# the model expects an input of shape(1, window_in, n_features  )

X_input = np.array(X_train[features][-365:].values)

X_input = X_input.reshape(1, window_in, n_features)



y_hat = model.predict([X_input, X_input, X_input, X_input, X_input, X_input], verbose=0)
# plot the prediction and validation

plt.plot(y_hat[0], label='predicted_power')



y_true = X_valid.day_power.values[-365:]

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

print('r2 score %.2f ' % r2_score(y_true_cumulative, y_predicted_cumulative))

print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))