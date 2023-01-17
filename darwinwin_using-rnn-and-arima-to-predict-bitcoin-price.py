# First step, import libraries and then dataset

import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt
# Import the dataset and encode the date

df = pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv")

df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date

group = df.groupby('date')

Real_Price = group['Weighted_Price'].mean()
# split data

prediction_days = 30

df_train= Real_Price[len(Real_Price)-prediction_days:]

df_test= Real_Price[:len(Real_Price)-prediction_days]
# Data preprocess

training_set = df_train.values

training_set = np.reshape(training_set, (len(training_set), 1))

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

training_set = sc.fit_transform(training_set)

X_train = training_set[0:len(training_set)-1]

y_train = training_set[1:len(training_set)]

X_train = np.reshape(X_train, (len(X_train), 1, 1))
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



# Initialising the RNN

regressor = Sequential()



# Adding the input layer and the LSTM layer

regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
test_set = df_test.values[1:]

sc = MinMaxScaler()

inputs = np.reshape(df_test.values[0:len(df_test)-1], (len(test_set), 1))

inputs = sc.transform(inputs)

inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted_BTC_price = regressor.predict(inputs)

predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
# Visualising the results

plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()  

plt.plot(test_set, color = 'red', label = 'Real BTC Price')

plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')

plt.title('BTC Price Prediction', fontsize=40)

df_test = df_test.reset_index()

x=df_test.index

labels = df_test['date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

plt.xlabel('Time', fontsize=40)

plt.ylabel('BTC Price(USD)', fontsize=40)

plt.legend(loc=2, prop={'size': 25})

plt.show()

# Import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

from scipy import stats

import statsmodels.api as sm

import warnings

from itertools import product

from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-poster')
# Load data

df = pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv")

df.head()
# Unix-time to 

df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')



# Resampling to daily frequency

df.index = df.Timestamp

df = df.resample('D').mean()



# Resampling to monthly frequency

df_month = df.resample('M').mean()



# Resampling to annual frequency

df_year = df.resample('A-DEC').mean()



# Resampling to quarterly frequency

df_Q = df.resample('Q-DEC').mean()
# PLOTS

fig = plt.figure(figsize=[15, 7])

plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)



plt.subplot(221)

plt.plot(df.Weighted_Price, '-', label='By Days')

plt.legend()



plt.subplot(222)

plt.plot(df_month.Weighted_Price, '-', label='By Months')

plt.legend()



plt.subplot(223)

plt.plot(df_Q.Weighted_Price, '-', label='By Quarters')

plt.legend()



plt.subplot(224)

plt.plot(df_year.Weighted_Price, '-', label='By Years')

plt.legend()



# plt.tight_layout()

plt.show()
plt.figure(figsize=[15,7])

sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()

print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])

plt.show()
# Box-Cox Transformations

df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)

print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
# Seasonal differentiation

df_month['prices_box_diff'] = df_month.Weighted_Price_box - df_month.Weighted_Price_box.shift(12)

print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])
# Regular differentiation

df_month['prices_box_diff2'] = df_month.prices_box_diff - df_month.prices_box_diff.shift(1)

plt.figure(figsize=(15,7))



# STL-decomposition

sm.tsa.seasonal_decompose(df_month.prices_box_diff2[13:]).plot()   

print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff2[13:])[1])



plt.show()
# Initial approximation of parameters

Qs = range(0, 2)

qs = range(0, 3)

Ps = range(0, 3)

ps = range(0, 3)

D=1

d=1

parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)



# Model Selection

results = []

best_aic = float("inf")

warnings.filterwarnings('ignore')

for param in parameters_list:

    try:

        model=sm.tsa.statespace.SARIMAX(df_month.Weighted_Price_box, order=(param[0], d, param[1]), 

                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)

    except ValueError:

        print('wrong parameters:', param)

        continue

    aic = model.aic

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param, model.aic])
# Best Models

result_table = pd.DataFrame(results)

result_table.columns = ['parameters', 'aic']

print(result_table.sort_values(by = 'aic', ascending=True).head())

print(best_model.summary())
# Inverse Box-Cox Transformation Function

def invboxcox(y,lmbda):

   if lmbda == 0:

      return(np.exp(y))

   else:

      return(np.exp(np.log(lmbda*y+1)/lmbda))
# Prediction

df_month2 = df_month[['Weighted_Price']]

date_list = [datetime(2017, 6, 30), datetime(2017, 7, 31), datetime(2017, 8, 31), datetime(2017, 9, 30), 

             datetime(2017, 10, 31), datetime(2017, 11, 30), datetime(2017, 12, 31), datetime(2018, 1, 31),

             datetime(2018, 1, 28)]

future = pd.DataFrame(index=date_list, columns= df_month.columns)

df_month2 = pd.concat([df_month2, future])

df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=75), lmbda)

plt.figure(figsize=(15,7))

df_month2.Weighted_Price.plot()

df_month2.forecast.plot(color='r', ls='--', label='Predicted Weighted_Price')

plt.legend()

plt.title('Bitcoin exchanges, by months')

plt.ylabel('mean USD')

plt.show()