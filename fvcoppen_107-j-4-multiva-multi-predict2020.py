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
solarpower.head(2), solarpower.tail(2)
# add a feature

solarpower['Gas_plus_Elek'] = solarpower.Gas_mxm + solarpower.Elec_kW
year1 = solarpower['2011-10-27':'2012-10-26'].copy()

year2 = solarpower['2012-10-27':'2013-10-26'].copy()

year3 = solarpower['2013-10-27':'2014-10-26'].copy()

year4 = solarpower['2014-10-27':'2015-10-26'].copy()

year5 = solarpower['2015-10-27':'2016-10-26'].copy()

year6 = solarpower['2016-10-27':'2017-10-26'].copy()

year7 = solarpower['2017-10-27':'2018-10-26'].copy()

year8 = solarpower['2018-10-27':'2019-10-26'].copy()

year9 = solarpower['2019-10-27':].copy()
from scipy import optimize



def test_func(x, a, b, c, d):

    return (a + b * np.cos(c + np.pi*(360 * x / 365)/180)) * np.cos(d) 
def plot_optimized(x_data, y_data, params):

    plt.figure(figsize=(4, 2))

    plt.scatter(x_data, y_data, c='y',label='Data')

    cosinus = test_func(x_data, params[0], params[1], params[2] , params[3] )

    plt.plot(x_data, cosinus, c='r',

         label='Fitted function')

    plt.legend(loc='best')

    plt.show()

    return cosinus
for year in [year1, year2, year3, year4, year5, year6, year7, year8, year9]:  

    x_data = year.index.dayofyear

    y_data = year['day_power'].values

    params, params_covariance = optimize.curve_fit(test_func, x_data, y_data)

    print(params)

    cosinus = plot_optimized(x_data, y_data, params)

    year['cosinus'] = cosinus
solarpower3 = pd.concat([year1,year2,year3,year4,year5,year6,year7,year8, year9], axis='rows')
solarpower3.head(3), solarpower3.tail(3)
X_train = solarpower3[:]

#X_valid = solarpower3['2018-10-27':] # is 365 days

#X_train.shape, X_valid.shape
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
# apply my_split_window on daily solar power with a window of 365 days (we do not make account for leap years)



window_in = 365

window_out = 365

features = ['day_power','Elec_kW', 'Gas_mxm', 'Gas_plus_Elek', 'cosinus']

y_series = X_train.day_power.values

X, y = my_split_window(np.array(X_train[features]) , y_series ,  window_in, window_out)

print('X.shape', X.shape, 'y.shape', y.shape)

# print a sample

for i in range(2):

    print(X[i][-2:], y[i][-2:])
# vector output model:

# model for univariate series input and prediction of  timestep vector

# we have an input shape = (number of windows, window_in) 

#  and we have a window size of one year (365 days)

# the output vector is of shape(number of window_out)

window_in = 365

window_out = 365

n_features = X.shape[2]

print('n_features',n_features)

features = ['day_power', 'Elec_kW', 'Gas_mxm', 'Gas_plus_Elek', 'cosinus']

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

epochs=12000

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

    name = '107_I_y_hat_10000e' + str(steps)

    y_hat_dict[name]=y_hat[0]

    file = name + 'range' + str(range1) + '.npy'

    np.save(file, y_hat[0])

    print('step', steps, 'done')
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
#y_true = X_valid.day_power.values



#plt.plot(y_true, label='true_power')

for key, value in y_hat_dict.items()  :

    plt.plot(value, label=key)

    plt.legend()

    #first_r2_score = r2_score(y_true, value) # Best possible score is 1.0 

    #first_mae = mean_absolute_error(y_true, value)

    #print('r2_score %.5f' % first_r2_score)

    #print('mae %.2f' % first_mae)

plt.legend()

plt.show()



#y_true_cumulative = cumulate(y_true)

#plt.plot(y_true_cumulative, label='true_power')

for key, value in y_hat_dict.items()  :

    y_predicted_cumulative = cumulate(value)

    plt.plot(y_predicted_cumulative, label='predicted_power')

    #true_cumulative_power_after_one_year = int(y_true_cumulative[-1])

    predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])

    #print('true cumulative power after one year:', true_cumulative_power_after_one_year)

    #print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)

    #acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year

    #acc_one_year = acc_one_year * 100

    #print('accuracy after one year: %.2f' %  acc_one_year,'%')

    #print('r2 score %.2f ' % r2_score(y_true_cumulative, y_predicted_cumulative))

    #print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))

    

plt.legend()

plt.show()