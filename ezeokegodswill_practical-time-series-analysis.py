# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading our data

df = pd.read_csv("/kaggle/input/household-power-consumption/household_power_consumption.txt", delimiter=';', infer_datetime_format=True, parse_dates={'datetime': [0, 1]}, index_col=['datetime'])
df.head()
# replace all "?" in the dataset with NaN

df.replace("?", "NaN", inplace=True)
# change our dataset type

df.astype(float)
# replace the NaN values with the numpy nan

df.replace({"NaN": np.nan}, inplace=True)
# check for number of Na Values in each column

df.isna().sum()
# fill the na values in each column with the mean of their respective columns



df.Global_active_power.fillna(df.Global_active_power.mean(), inplace=True)

df.Sub_metering_3.fillna(df.Sub_metering_3.mean(), inplace=True)

df.Global_reactive_power.fillna(df.Global_reactive_power.mean(), inplace=True)

df.Global_intensity.fillna(df.Global_intensity.mean(), inplace=True)

df.Voltage.fillna(df.Voltage.mean(), inplace=True)

df.Sub_metering_1.fillna(df.Sub_metering_1.mean(), inplace=True)

df.Sub_metering_2.fillna(df.Sub_metering_2.mean(), inplace=True)
# check for na values again to be sure

df.isna().sum()
# calculating total energy consumed every minute

energy_consumed = ((df["Global_active_power"]*1000)/60 - df.Sub_metering_1 - df.Sub_metering_2 - df.Sub_metering_3)
# create a new column in our dataset and assign the energy consumed values to it

df["energy_consumed"] = energy_consumed

df.head()
#Upsample to daily data points

df_daily = df.resample('D').mean()
df_daily.head()
df_daily.tail()
import matplotlib.pyplot as plt
# make a plot of our global active power against every month

plt.figure(figsize=(8,7))

plt.plot(df_daily.index, df_daily.Global_active_power)

plt.xlabel('Day')

plt.ylabel('GAP') 
# get the pearson correlation of our dataset

df_daily.corr(method ='pearson')
df_MA = df_daily.copy()

MA = df_MA['energy_consumed'].rolling(12).mean()
plt.figure(figsize=(15 , 10))

plt.plot(df_daily.index, df_daily.energy_consumed, '--' , marker= '*')

plt.plot(MA, color="red")

plt.grid()

plt.xlabel('Day')

plt.ylabel('energy_consumed') 
import statsmodels.api as sm

from pylab import rcParams



rcParams['figure.figsize'] = 15, 8

decompose_series = sm.tsa.seasonal_decompose(df_daily['energy_consumed'], model= 'additive')



decompose_series.plot()

plt.show()

from statsmodels.tsa.stattools import adfuller



adf_result = adfuller(df_daily['energy_consumed'])

print(f'ADF Statistic: {adf_result[0]}')

print(f'p-value: {adf_result[1]}')

print(f'No. of lags used: {adf_result[2]}')

print(f'No. of observations used : {adf_result[3]}')

print('Critical Values:')





for k, v in adf_result[4].items():

    print(f'{k} : {v}')
# get only the energy_consumed column for our prediction

dfx = df_daily["energy_consumed"]
dfx
from pandas import DataFrame

# create a dataframe from the list returned

dfy = DataFrame(dfx)
# reset index of our dataset

dfy.reset_index(inplace=True)
# renaming the columns to work with our prohpet model

dfy = dfy.rename(columns={ "datetime" : "ds" , "energy_consumed" : "y" })
# have a quick look at our data

dfy
df_log = np.log(dfx)

df_diff = df_log.diff(periods = 1)



plt.plot(df_diff.index, df_diff, '-')

plt.plot(df_diff.rolling(12).mean(), color= 'red') 
df_log
df_diff.fillna(0, inplace=True)
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



#ACF

plot_acf(df_diff, lags = range( 0 , 20 ))

plt.show()



#PACF

plot_pacf(df_diff, lags = range( 0 , 20 ))

plt.show() 
from statsmodels.tsa.arima_model import ARIMA

ARIMA_model = ARIMA(df_diff, order=( 2 , 0 , 1 )) 

ARIMA_results = ARIMA_model.fit(disp=0)

plt.plot(df_diff)

plt.plot(ARIMA_results.fittedvalues, color= 'red' ) 
# creating and fit our model on the data (dfy)



from fbprophet import Prophet

model = Prophet()

model.fit(dfy)
#predict for the next 10 months



future = model.make_future_dataframe(periods=10, freq= 'M')

forecast = model.predict(future)

forecast.head()

forecast[['ds' , 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]

# take a look at our forecast/predicted data

forecast
#yhat is the prediction while yhat_lower and yhat_upper are the upper and lower boundaries 

model.plot(forecast)

plt.show()
# create a plot for our trend



plt.figure(figsize=(12 , 7))

plt.plot(forecast.ds, forecast.trend)

plt.grid()

plt.xlabel('ds')

plt.ylabel('trend')
# create a plot for the yearly values in our forecast data



plt.figure(figsize=(12 , 7))

plt.plot(forecast.ds, forecast.yearly)

plt.grid()

plt.xlabel('ds')

plt.ylabel('yearly')
# getting the expected and predicted values

expected = dfy.y

predictions = forecast.yhat
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]

print('Forecast Errors: %s' % forecast_errors)
bias = sum(forecast_errors) * 1.0/len(expected)

print('Bias: %f' % bias)