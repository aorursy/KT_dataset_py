import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

import tensorflow as tf

import keras

print('tf version:',tf.__version__,'\n' ,'keras version:',keras.__version__,'\n' ,'numpy version:',np.__version__)
# load previous prediction results

predicted_data = pd.read_hdf('../input/104-sol-elec-gas-2-c-multivariate-parallel-series/predicted_data4.hdf5')


solarpower = pd.read_csv("../input/solarpanelspower/PV_Elec_Gas2.csv",header = None,skiprows=1 ,names = ['date','cum_power','Elec_kW', 

                                                                            'Gas_mxm'], sep=',',usecols = [0,1,2,3],

                     

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



def my_split_window(array, window):

    '''

    the array has the columns (features) that we use as input.

    Returns array X with the features windowed in shape (number of windows, window, n_features)

    and array y with n_features

    '''

    X = []

    y = []

    n_steps = len(array) - window

    for step in range(n_steps):

        X_w = []

        for i in range(window):

            X_w.append(array[step + i])

        X.append(X_w)

        y.append(array[step + window])

    X = np.array(X)

    y = np.array(y)

    return X, y
# test my_split_window

df = pd.DataFrame()

df['feature1'] = [10,20,30,40,50,60,70,80,90]

df['feature2'] = [11,21,31,41,51,61,71,81,91]

df['feature3'] = [26, 46, 66, 86, 106, 126, 146, 166, 186]

features_test = ['feature1','feature2','feature3']

array = np.array(df[features_test])

# print(array[:3])

window = 3

X_, y_ = my_split_window(array, window)

X_, y_



X_.shape, y_.shape
# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)



window = 365

features = X_train.columns.values

array = np.array(X_train[features])

X, y = my_split_window(array,  window)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i])
# model for Multiple parallel series input and prediction of one timestep parallel features

# Each output series can be handled by a separate output CNN model (multi-output-cnn)

# we have an input shape = (number of windows, window, n_features) 

#  and we have a window size of one year (365 days) 



n_features = X.shape[2]

window = 365

# define model

def input_model(window, n_features):

    visible = tf.keras.layers.Input(shape=(window, n_features))

    cnn = tf.keras.layers.BatchNormalization()(visible)

    cnn = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu')(cnn)

    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)

    cnn = tf.keras.layers.Flatten()(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)

    cnn_model = tf.keras.layers.Dense(50, activation='relu')(cnn)

    return visible, cnn_model



# we define one output layer for each feature

def output_3f_model(visible, cnn_model):

    output1 = tf.keras.layers.Dense(1)(cnn_model)

    output2 = tf.keras.layers.Dense(1)(cnn_model)

    output3 = tf.keras.layers.Dense(1)(cnn_model)

    model = tf.keras.Model(inputs=visible, outputs = [output1, output2, output3])

    return model



visible, cnn_model = input_model(window, n_features)

model = output_3f_model(visible, cnn_model)



model.compile(optimizer='adam', loss='mae')

# separate output

y1 = y[:, 0].reshape((y.shape[0], 1))

y2 = y[:, 1].reshape((y.shape[0], 1))

y3 = y[:, 2].reshape((y.shape[0], 1))



# fit model

history = model.fit(X, [y1, y2, y3], epochs=200, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting next year based on data of X_valid to see if model is ok

# the model expects an input of shape(n_time steps = window size, n_features)

y_hat = []

X_input =  np.array(X_train[-365:]) #  next value based on data of last year

X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1]) # input must have 3 dimensions

x_input=X_input

for i in range(365):

    new_x = np.array(X_valid.iloc[i])

    new_x = new_x.reshape(1, x_input.shape[0], x_input.shape[2])

    x_input = np.concatenate((x_input[:, -364:], new_x), axis=1)

    y_hat.append(model.predict(x_input, verbose=0))





y_hat = np.array(y_hat)



y_hat = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])


plt.plot(y_hat[:,2], label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat[:,2]) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat[:,2])

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

y_predicted_cumulative = cumulate(y_hat[:,2])



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
X_train = X_train.copy()

X_valid = X_valid.copy()

X_train['Gas_plus_Elek'] = X_train.Gas_mxm + X_train.Elec_kW

X_valid['Gas_plus_Elek'] = X_valid.Gas_mxm + X_valid.Elec_kW
# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)



window = 365

features = X_train.columns.values

array = np.array(X_train[features])

X, y = my_split_window(array,  window)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i])
# model for Multiple parallel series input and prediction of one timestep parallel features

# Each output series can be handled by a separate output CNN model (multi-output-cnn)

# we have an input shape = (number of windows, window, n_features) 

#  and we have a window size of one year (365 days) 



n_features = X.shape[2]

window = 365

# define model

visible, cnn_model = input_model(window, n_features)



# we define one output layer for each feature

def output_4f_model(visible, cnn_model):

    output1 = tf.keras.layers.Dense(1)(cnn_model)

    output2 = tf.keras.layers.Dense(1)(cnn_model)

    output3 = tf.keras.layers.Dense(1)(cnn_model)

    output4 = tf.keras.layers.Dense(1)(cnn_model)

    model = tf.keras.Model(inputs=visible, outputs = [output1, output2, output3, output4])

    return model



model = output_4f_model(visible, cnn_model)



model.compile(optimizer='adam', loss='mae')

# separate output

y1 = y[:, 0].reshape((y.shape[0], 1))

y2 = y[:, 1].reshape((y.shape[0], 1))

y3 = y[:, 2].reshape((y.shape[0], 1))

y4 = y[:, 3].reshape((y.shape[0], 1))

# fit model

history = model.fit(X, [y1, y2, y3, y4], epochs=200, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting next year

# the model expects an input of shape(n_time steps = window size, n_features)

y_hat = []

X_input =  np.array(X_train[-365:]) #  next value based on data of last year

X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1]) # input must have 3 dimensions

x_input=X_input

for i in range(365):

    new_x = np.array(X_valid.iloc[i])

    new_x = new_x.reshape(1, x_input.shape[0], x_input.shape[2])

    x_input = np.concatenate((x_input[:, -364:], new_x), axis=1)

    y_hat.append(model.predict(x_input, verbose=0))

    

y_hat = np.array(y_hat)

y_hat = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])


plt.plot(y_hat[:,2], label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat[:,2]) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat[:,2])

print('r2_score %.5f' % first_r2_score)

print('mae %.2f' % first_mae)
y_true_cumulative = cumulate(y_true)

y_predicted_cumulative = cumulate(y_hat[:,2])



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
# prediction based on prediction
# model for Multiple parallel series input and prediction of one timestep parallel features

# Each output series can be handled by a separate output CNN model (multi-output-cnn)

# we have an input shape = (number of windows, window, n_features) 

#  and we have a window size of one year (365 days) 



n_features = X.shape[2]

window = 365

# define model

visible, cnn_model = input_model(window, n_features)



# we define one output layer for each feature

def output_4f_model(visible, cnn_model):

    output1 = tf.keras.layers.Dense(1)(cnn_model)

    output2 = tf.keras.layers.Dense(1)(cnn_model)

    output3 = tf.keras.layers.Dense(1)(cnn_model)

    output4 = tf.keras.layers.Dense(1)(cnn_model)

    model = tf.keras.Model(inputs=visible, outputs = [output1, output2, output3, output4])

    return model



model = output_4f_model(visible, cnn_model)



model.compile(optimizer='adam', loss='mae')

# separate output

y1 = y[:, 0].reshape((y.shape[0], 1))

y2 = y[:, 1].reshape((y.shape[0], 1))

y3 = y[:, 2].reshape((y.shape[0], 1))

y4 = y[:, 3].reshape((y.shape[0], 1))

# fit model

history = model.fit(X, [y1, y2, y3, y4], epochs=5000, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predict next year based on prediction of next step

# predicting next year

# the model expects an input of shape(1, n_time steps = window size, n_features)

y_hat = []

X_input =  np.array(X_train[-365:]) #  next last value is predicted value

X_input = X_input.reshape(1, window, n_features) # input must have 3 dimensions

for i in range(365):

    y_hat.append((model.predict(X_input, verbose=0)))

    #print(np.array(y_hat).shape)

    new_X = np.array(y_hat[i])[:,0,0]

    #print(new_X)

    new_X = new_X.reshape(1, 1,new_X.shape[0])

    X_input = np.concatenate((X_input[:, -364:], new_X), axis=1)

    X_input = X_input.reshape(1, window, n_features)

y_hat = np.array(y_hat)


plt.plot(y_hat[:,2,0,0], label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat[:,2,0,0]) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat[:,2,0,0])

print('r2_score %.5f' % first_r2_score)

print('mae %.2f' % first_mae)
y_true_cumulative = cumulate(y_true)

y_predicted_cumulative = cumulate(y_hat[:,2,0,0])



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
predicted_data['105_4f_CNN_multi_ouput_5000epochs'] = y_hat[:,2,0,0]

predicted_data.to_hdf('predicted_data5.hdf5',key='predicted_data', table='true',mode='a')




