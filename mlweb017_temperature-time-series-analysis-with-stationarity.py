import pandas as pd

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose



%matplotlib inline
temp = pd.read_csv('../input/GlobalLandTemperaturesByState.csv')
print (temp.head())

print('')

print ('Dtypes')

print (temp.dtypes)

print('')

print ('Shape')

print (temp.shape)
# Counts the number of null values in each column

temp.isnull().sum()
# Creating a new column indicating if we have a null value in the Avg_temp column (1) or not (0)

temp['Have_temp_data'] = temp['AverageTemperature'].apply(lambda x: 1 if not pd.isnull(x) else 0)

temp.head()
# Verifying all null values have 0 

temp['Have_temp_data'].value_counts()
# Null values by country

temp['Have_temp_data'].groupby(temp['Country']).value_counts()
# I want to rename the column names so they are more concise

temp.rename(columns={'dt':'Date', 'AverageTemperature':'Avg_temp', 'AverageTemperatureUncertainty':'Temp_confidence_interval'}, inplace=True)

temp.head()
# Convert the date column to datetime series

temp['Date'] = pd.to_datetime(temp['Date'])

temp.set_index('Date', inplace=True)

temp.index
# I want to extract just the year from the date column and create it's own column.

temp['Year'] = temp.index.year

temp.head()
# Reviewing if there's null values in the last years of the data grouped by Year

temp['Have_temp_data'].groupby(temp.index.year).value_counts().tail(45)
# Statistical information by column

temp.describe()
# Filtering by years 1970-2013 because these didn't have many null values

recent_temp = temp.loc['1970':'2013']

recent_temp.head()
#  Statistical information by country

recent_temp.groupby('Country').describe()
# Shows the average temperature by country in descending order

recent_temp[['Country','Avg_temp']].groupby(['Country']).mean().sort_values('Avg_temp',ascending=False)
recent_temp[['Avg_temp']].plot(kind='line',title='Temperature Changes from 1970-2013',figsize=(12,6))
# Resampling annual averages 

temp_resamp = recent_temp[['Avg_temp']].resample('A').mean()



# Temperature graph 

temp_resamp.plot(title='Temperature Changes from 1970-2013',figsize=(8,5))

plt.ylabel('Temperature',fontsize=12)

plt.xlabel('Year',fontsize=12)

plt.legend()



plt.tight_layout()
# Dickey-Fuller test

from statsmodels.tsa.stattools import adfuller



print ('Dickey-Fuller Test Results:')

dftest = adfuller(temp_resamp.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
# Decomposing the data

temp_decomp = seasonal_decompose(temp_resamp, freq=3)  



# Extracting the components

trend = temp_decomp.trend

seasonal = temp_decomp.seasonal

residual = temp_decomp.resid



# Plotting the original time series

plt.subplot(411)

plt.plot(temp_resamp)

plt.xlabel('Original')

plt.figure(figsize=(6,4))



# Plotting the trend component

plt.subplot(412)

plt.plot(trend)

plt.xlabel('Trend')

plt.figure(figsize=(6,4))



# Plotting the seasonal component

plt.subplot(413)

plt.plot(seasonal)

plt.xlabel('Seasonal')

plt.figure(figsize=(6,4))



# Plotting the residual component

plt.subplot(414)

plt.plot(residual)

plt.xlabel('Residual')

plt.figure(figsize=(6,4))



plt.tight_layout()
# Graphing just the trend line 

trend.plot(title='Temperature Trend Line',figsize=(8,4)) 



# Graph labels

plt.xlabel('Year',fontsize=12)

plt.ylabel('Temperature',fontsize=12)



plt.tight_layout()
# Rolling mean 

temp_rol_mean = temp_resamp.rolling(window=3, center=True).mean()



# Exponentially weighted mean 

temp_ewm = temp_resamp.ewm(span=3).mean()



# Rolling standard deviation 

temp_rol_std = temp_resamp.rolling(window=3, center=True).std()



# Creating subplots next to each other

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))



# Temperature graph with rolling mean and exponentially weighted mean

ax1.plot(temp_resamp,label='Original')

ax1.plot(temp_rol_mean,label='Rolling Mean')

ax1.plot(temp_ewm, label='Exponentially Weighted Mean')

ax1.set_title('Temperature Changes from 1970-2013',fontsize=14)

ax1.set_ylabel('Temperature',fontsize=12)

ax1.set_xlabel('Year',fontsize=12)

ax1.legend()



# Temperature graph with rolling STD 

ax2.plot(temp_rol_std,label='Rolling STD')

ax2.set_title('Temperature Changes from 1970-2013',fontsize=14)

ax2.set_ylabel('Temperature',fontsize=12)

ax2.set_xlabel('Year',fontsize=12)

ax2.legend()



plt.tight_layout()

plt.show()
# Dickey-Fuller test 

temp_rol_mean.dropna(inplace=True)

temp_ewm.dropna(inplace=True)

print ('Dickey-Fuller Test for the Rolling Mean:')

dftest = adfuller(temp_rol_mean.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)

print ('')

print ('Dickey-Fuller Test for the Exponentially Weighted Mean:')

dftest = adfuller(temp_ewm.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
# Difference between the original and the rolling mean 

diff_rol_mean = temp_resamp - temp_rol_mean

diff_rol_mean.dropna(inplace=True)

diff_rol_mean.head()
# Difference between the original and the exponentially weighted mean

diff_ewm = temp_resamp - temp_ewm

diff_ewm.dropna(inplace=True)

diff_ewm.head()
# Rolling mean of the difference

temp_rol_mean_diff = diff_rol_mean.rolling(window=3, center=True).mean()



# Expotentially weighted mean of the difference

temp_ewm_diff = diff_ewm.ewm(span=3).mean()



# Creating subplots next to each other

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))



# Difference graph with the rolling mean

ax1.plot(diff_rol_mean,label='Original')

ax1.plot(temp_rol_mean_diff,label='Rolling Mean')

ax1.set_title('Temperature Changes from 1970-2013',fontsize=14)

ax1.set_ylabel('Temperature',fontsize=12)

ax1.set_xlabel('Year',fontsize=12)

ax1.legend()



# Difference graph with the exponentially weighted mean

ax2.plot(diff_ewm,label='Original')

ax2.plot(temp_ewm_diff,label='Exponentially Weighted Mean')

ax2.set_title('Temperature Changes from 1970-2013',fontsize=14)

ax2.set_ylabel('Temperature',fontsize=12)

ax2.set_xlabel('Year',fontsize=12)

ax2.legend()



plt.tight_layout()
# Dickey-Fuller test 

print ('Dickey-Fuller Test for the Difference between the Original and Rolling Mean:')

dftest = adfuller(diff_rol_mean.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)

print ('')

print ('Dickey-Fuller Test for the Difference between the Original and Exponentially Weighted Mean:')

dftest = adfuller(diff_ewm.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
# Shifting forward by 1 year

temp_shift1 = temp_resamp.shift(1)

temp_shift1.head()
# Difference between the original and time series shifted by 1 year 

shift1_diff = temp_resamp - temp_shift1

shift1_diff.dropna(inplace=True)



# Rolling mean 

temp_shift1_diff_rol_mean = shift1_diff.rolling(window=3, center=True).mean()



# Exponentially weighted mean 

temp_shift1_diff_ewm = shift1_diff.ewm(span=3).mean()



# Rolling standard deviation 

temp_shift1_diff_rol_std = shift1_diff.rolling(window=3, center=True).std()



# Creating subplots next to each other

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))



# Temperature graph 

ax1.plot(shift1_diff,label='Original')

ax1.plot(temp_shift1_diff_rol_mean,label='Rolling Mean')

ax1.plot(temp_shift1_diff_ewm,label='Exponentially Weighted Mean')

ax1.set_title('Shifted By 1 Year Temperature Changes from 1970-2013',fontsize=14)

ax1.set_ylabel('Temperature',fontsize=12)

ax1.set_xlabel('Year',fontsize=12)

ax1.legend()



# Temperature Rolling STD graph

ax2.plot(temp_shift1_diff_rol_std)

ax2.set_title('Shifted By 1 Year Rolling Standard Deviation',fontsize=14)

ax2.set_ylabel('Temperature',fontsize=12)

ax2.set_xlabel('Year',fontsize=12)



plt.tight_layout()

plt.show()
# Dickey-Fuller test 

print ('Dickey-Fuller Test for Difference between the Original and Shifted by 1 Year:')

dftest = adfuller(shift1_diff.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
# Drop N/A values

residual.dropna(inplace=True)



# Residuals rolling mean

resid_rol_mean = residual.rolling(window=3).mean()



# Residuals exponentially weighted mean

resid_ewm = residual.ewm(span=3).mean()



# Residuals rolling standard deviation 

resid_rol_std = residual.rolling(window=3, center=True).std()



# Creating subplots next to each other

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))



# Temperature graph with residual rolling mean and exponentially weighted mean

ax1.plot(residual,label='Original')

ax1.plot(resid_rol_mean,label='Rolling Mean')

ax1.plot(resid_ewm, label='Exponentially Weighted Mean')

ax1.set_title('Residuals',fontsize=14)

ax1.set_ylabel('Temperature',fontsize=12)

ax1.set_xlabel('Year',fontsize=12)

ax1.legend()



# Temperature graph with residual rolling STD 

ax2.plot(resid_rol_std,label='Rolling STD')

ax2.set_title('Residuals Rolling STD',fontsize=14)

ax2.set_ylabel('Temperature',fontsize=12)

ax2.set_xlabel('Year',fontsize=12)

ax2.legend()



plt.tight_layout()

plt.show()
# Dickey-Fuller test 

print ('Dickey-Fuller Test for the Residuals:')

dftest = adfuller(residual.iloc[:,0].values, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from matplotlib import pyplot



# Plotting the autocorrelation and partial autocorrelation graphs

pyplot.figure(figsize=(10,5))

pyplot.subplot(211)

plot_acf(temp_resamp, ax=pyplot.gca())

pyplot.subplot(212)

plot_pacf(temp_resamp, ax=pyplot.gca())

pyplot.show()