# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from pandas import read_csv

from pandas import DataFrame

from pandas import Grouper

from matplotlib import pyplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/industry-production-index/Industry_Trends.csv", header=0, index_col=0, parse_dates=True, squeeze=True)

print(df.head())

print('''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Industrial Production Index (INDPRO) is an economic indicator that measures real output for all facilities located in the United States manufacturing, mining, and electric, and gas utilities (excluding those in U.S. territories). Since 1997, the Industrial Production Index has been determined from 312 individual series based on the 2007 North American Industrial Classification System (NAICS) codes. These individual series are classified in two ways (1) market groups and (2) industry groups.

(1) The Board of Governors defines markets groups as products (aggregates of final products) and materials (inputs used in the manufacture of products). Consumer goods and business equipment can be examples of market groups. "Industry groups are defined as three digit NAICS industries and aggregates of these industries such as durable and nondurable manufacturing, mining, and utilities."(1)(2)

The index is compiled on a monthly basis to bring attention to short- term changes in industrial production,. It measures movements in production output and highlights structural developments in the economy. (1) Growth in the production index from month to month is an indicator of growth in the industry.

For more information regarding the Industrial Production and Capacity Utilization index, see the explanatory notes issued by the Board of Governors.



References

(1) Board of Governors of the Federal Reserve System. "Industrial Production and Capacity Utilization." Statistical release G.17; May 2013.

(2) For recent reports on market and industry groups, please visit the Board of Governors.''')
#plot historical data

plt.style.use('seaborn-muted')

%matplotlib inline



df.plot.area(title='Total Industrial Output',label = 'Original',color='deepskyblue',linewidth=2.5,figsize=(20, 6),fontsize=16);



plt.xlabel('Year',fontsize=20)

plt.ylabel('Industrial Output',fontsize=20)





#subplot layover

subdate = df[['1/1/2000','2/1/2020']]

subdate.plot(subplots=True,  linestyle='--',color='r', linewidth=3,figsize=(20, 6),fontsize=16)

df.plot.line(subplots=True,  linestyle='-',color='k', linewidth=2,figsize=(20, 6),fontsize=16);

plt.show()



rolling_mean = df.rolling(window = 12).mean()

rolling_std = df.rolling(window = 12).std()



plt.plot(rolling_mean,  color = 'red', label = 'Rolling Mean')

plt.plot(rolling_std, color = 'black', label = 'Rolling Std')

plt.legend(loc = 'best')

#Autocorrelation

from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf

from matplotlib.pyplot import figure





autocorrelation_plot(df)

pyplot.show()
#AutoRegression Train/Test

from statsmodels.tsa.ar_model import AutoReg

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller

from math import sqrt





X = df.values

train, test = X[1:len(X)-7], X[len(X)-7:]



# train autoregression

model = AutoReg(train, lags=10)

model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)



# make predictions

predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

for i in range(len(predictions)):

	print('predicted=%f, expected=%f' % (predictions[i], test[i]))

rmse = sqrt(mean_squared_error(test, predictions))

print('Test RMSE: %.3f' % rmse)



# plot results

pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()
#Making time series stationary 

df_log = np.log(df)

def get_stationarity(timeseries):

    

    # rolling statistics

    rolling_mean = timeseries.rolling(window=12).mean()

    rolling_std = timeseries.rolling(window=12).std()

    

    # rolling statistics plot

    original = plt.plot(timeseries, color='blue', label='Original')

    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')

    std = plt.plot(rolling_std, color='black', label='Rolling Std')

    plt.legend(loc='best')

    

    plt.show(block=False)

    

        # Dickey–Fuller test:

    result = adfuller(timeseries)

    print('ADF Statistic: {}'.format(result[0]))

    print('p-value: {}'.format(result[1]))

    print('Critical Values:')

    for key, value in result[4].items():

        print('\t{}: {}'.format(key, value))



#Subtract Rolling Mean        



rolling_mean = df_log.rolling(window=12).mean()

df_log_minus_mean = df_log - rolling_mean

df_log_minus_mean.dropna(inplace=True)

get_stationarity(df_log_minus_mean)

print('Subtract Rolling Mean')



#Exponential Decay 

rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()

df_log_exp_decay = df_log - rolling_mean_exp_decay

df_log_exp_decay.dropna(inplace=True)

get_stationarity(df_log_exp_decay)

print('Exponential Decay')



#Time Shifting, subtract every point by the one that preceded it.

df_log_shift = df_log - df_log.shift()

df_log_shift.dropna(inplace=True)

get_stationarity(df_log_shift)

print('Time Shifting, subtract every point by the one that preceded it')
#Building ARIMA model

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA



decomposition = seasonal_decompose(df_log) 

model = ARIMA(df_log, order=(2,1,2))

results = model.fit(disp=-1)

plt.plot(df_log_shift)

plt.plot(results.fittedvalues, color='red')



predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(df_log.iloc[0], index=df_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(df)

plt.plot(predictions_ARIMA)



results.plot_predict()