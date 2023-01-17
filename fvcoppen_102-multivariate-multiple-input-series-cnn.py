import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

import tensorflow as tf

import keras

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
solarpower.head(2), solarpower.tail(2)
X_train = solarpower[:'2018-10-28']

X_valid = solarpower['2018-10-29':'2019-10-28'] # is 365 days

X_train.shape, X_valid.shape
X_train.tail(2), X_valid.head(2)
# we devide the series into multiple input and output patterns



def my_split_window(dataframe, out_sequence, window):

    '''

    the Pandas dataframe has the columns (features) that we use as input (X).

    out_sequence is the time series that matches the input

    Returns array X with the features windowed in shape (number of windows, window, n_features)

    and array y

    '''

    X = []

    y = []

    n_steps = len(out_sequence) - window + 1

    for step in range(n_steps):

        X_w = []

        for i in range(window):

            X_w.append(dataframe.iloc[step + i])

        X.append(X_w)

        y.append(out_sequence[step + window -1])

    X = np.array(X)

    y = np.array(y)

    return X, y
# test my_split_window

df = pd.DataFrame()

df['feature1'] = [10,20,30,40,50,60,70,80,90]

df['feature2'] = [11,21,31,41,51,61,71,81,91]

out_sequence = [26, 46, 66, 86, 106, 126, 146, 166, 186]

window = 3

X_, y_ = my_split_window(df, out_sequence, window)

X_, y_

# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)

# the output series is the daily solar power

X_features = [ 'Elec_kW' , 'Gas_mxm']

X_train_input = X_train[ X_features]

out_sequence = X_train.day_power.values

window = 365

X, y = my_split_window(X_train_input, out_sequence,  window)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i])
# define model

def my_CNN_model(window, n_features):

    # define model

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', 

                                 input_shape=(window, n_features)))

    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mae')

    return model

#tf.keras.backend.clear_session()



# we have an input shape = (number of windows, window, n_features) 

#  and we have a window size of one year (365 days) 

# we do not have to reshape for the 1D CNN



epochs = 200

n_features = X.shape[2]

window = 365

model_solar = my_CNN_model(window, n_features)

model_solar.compile(optimizer='adam', loss='mae')



# fit model

history_solar = model_solar.fit(X, y, epochs=epochs, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history_solar.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# predicting validation year but with validation values for Electricity and Gas to verify quality of the model

# if we want prediction into the future year where we have no data for Elec and Gas then we have to train models

# that can predict those features

# the model expects an input of shape(n_time steps = window size, n_features)

y_hat = []

features = ['Elec_kW' , 'Gas_mxm']

X_input_solar =  np.array(X_train[features][-365:]) #  next value based on data of last year

X_input_solar = X_input_solar.reshape(1, X_input_solar.shape[0], X_input_solar.shape[1]) # input must have 3 dimensions

X_input_solar = X_input_solar

for i in range(365):

    new_x = np.array(X_valid[features].iloc[i])

    new_x = new_x.reshape(1, 1, X_input_solar.shape[2])

    X_input_solar = np.concatenate((X_input_solar[:, (-364):], new_x), axis=1)

    y_hat.append((model_solar.predict(X_input_solar, verbose=0).ravel())[0])


plt.plot(y_hat, label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat)

print('r2_score %.2f' % first_r2_score)

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

y_predicted_cumulative = cumulate(y_hat)



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
# save the first results



results = pd.DataFrame()

results['102_3_features_on_validation'] = y_hat

results.to_csv('102_3_features_on_validation.csv')
X_train = X_train.copy()

X_valid = X_valid.copy()

X_train['Gas_plus_Elek'] = X_train.Gas_mxm + X_train.Elec_kW

X_valid['Gas_plus_Elek'] = X_valid.Gas_mxm + X_valid.Elec_kW
# apply my_split_window on dayly solar power with a window of 365 days (we do not make account for leap years)

# the input series is the daily solar power

X_features = ['day_power','Elec_kW' , 'Gas_mxm', 'Gas_plus_Elek']

X_train_input = X_train[ X_features]

out_sequence = X_train.day_power.values

window = 365

X, y = my_split_window(X_train_input, out_sequence,  window)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i])
# apply my_split_window on dayly solar power with a window of 365 days (we do not make account for leap years)

# the input series is the daily Elec_kW

X_features_elec = ['day_power' , 'Gas_mxm']

X_train_input = X_train[ X_features_elec]

out_sequence = X_train.Elec_kW.values

window = 365

X, y = my_split_window(X_train_input, out_sequence,  window)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i])



#tf.keras.backend.clear_session()

# we have an input shape = (number of windows, window, n_features) 

#  and we have a window size of one year (365 days) 

# we do not have to reshape for the 1D CNN



n_features = X.shape[2]



model_elec = my_CNN_model(window, n_features)

model_elec.compile(optimizer='adam', loss='mae')

epochs = 600

# fit model

history_elec = model_elec.fit(X, y, epochs=epochs, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history_elec.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
# apply my_split_window on dayly solar power with a window of 365 days (we do not make account for leap years)

# the output series is the daily Gas

X_features_gas = ['day_power' , 'Elec_kW']

X_train_input = X_train[ X_features_gas]

out_sequence = X_train.Gas_mxm.values

window = 365

X, y = my_split_window(X_train_input, out_sequence,  window)

# print a sample

for i in range(3):

    print(X[i][-2:], y[i])
#tf.keras.backend.clear_session()

# we have an input shape = (number of windows, window, n_features) 

#  and we have a window size of one year (365 days) 

# we do not have to reshape for the 1D CNN



n_features = X.shape[2]



model_gas = my_CNN_model(window, n_features)

model_gas.compile(optimizer='adam', loss='mae')



# fit model

history_gas = model_gas.fit(X, y, epochs=epochs, verbose=0)



# graph of the loss shows convergence

import matplotlib.pyplot as plt

plt.plot(history_gas.history['loss'])

plt.title('loss')

plt.xlabel('epochs')

plt.show()
features_solar = ['Elec_kW' , 'Gas_mxm']

X_input_solar =  np.array(X_train[features_solar][-365:]) #  next value based on data of last year

X_input_solar = X_input_solar.reshape(1, X_input_solar.shape[0], X_input_solar.shape[1]) # input must have 3 dimensions

x_input_solar = X_input_solar



features_gas = ['Elec_kW' , 'day_power']

X_input_gas =  np.array(X_train[features_gas][-365:]) #  next value based on data of last year

X_input_gas = X_input_gas.reshape(1, X_input_gas.shape[0], X_input_gas.shape[1]) # input must have 3 dimensions

x_input_gas = X_input_gas



features_elec = ['day_power' , 'Gas_mxm']

X_input_elec =  np.array(X_train[features_elec][-365:]) #  next value based on data of last year

X_input_elec = X_input_elec.reshape(1, X_input_elec.shape[0], X_input_elec.shape[1]) # input must have 3 dimensions

x_input_elec = X_input_elec



y_hat_solar = ((model_solar.predict(x_input_solar, verbose=0).ravel())[0])

y_hat_gas = ((model_gas.predict(x_input_gas, verbose=0).ravel())[0])

y_hat_elec = ((model_elec.predict(x_input_elec, verbose=0).ravel())[0])

print('s', y_hat_solar, 'g', y_hat_gas, 'e', y_hat_elec)

y_hat_solar = []

y_hat_gas = []

y_hat_elec = []

for i in range(365):

    y_hat_solar.append((model_solar.predict(x_input_solar, verbose=0).ravel())[0])

    y_hat_gas.append((model_gas.predict(x_input_gas, verbose=0).ravel())[0])    

    y_hat_elec.append((model_elec.predict(x_input_elec, verbose=0).ravel())[0])

    

    new_x_solar = np.array((y_hat_elec[i], y_hat_gas[i]) )

    new_x_solar = new_x_solar.reshape(1, 1, 2)

    x_input_solar = np.concatenate((x_input_solar[:, (-364):], new_x_solar), axis=1)

    

    new_x_gas = np.array((y_hat_elec[i], y_hat_solar[i]) )

    new_x_gas = new_x_gas.reshape(1, 1, 2)

    x_input_gas = np.concatenate((x_input_gas[:, (-364):], new_x_gas), axis=1)

    

    new_x_elec = np.array((y_hat_solar[i], y_hat_gas[i]))

    new_x_elec = new_x_elec.reshape(1, 1, 2)

    x_input_elec = np.concatenate((x_input_elec[:, (-364):], new_x_elec), axis=1)

       


plt.plot(y_hat_solar, label='predicted_power')



y_true = X_valid.day_power.values

plt.plot(y_true, label='true_power')

plt.legend()

plt.show()
first_r2_score = r2_score(y_true, y_hat_solar) # Best possible score is 1.0 

first_mae = mean_absolute_error(y_true, y_hat_solar)

print('r2_score %.2f' % first_r2_score)

print('mae %.2f' % first_mae)
y_true_cumulative = cumulate(y_true)

y_predicted_cumulative = cumulate(y_hat_solar)



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
results['102_4_features_on_validation'] = y_hat_solar

results.to_csv('102_4_features_on_validation.csv')