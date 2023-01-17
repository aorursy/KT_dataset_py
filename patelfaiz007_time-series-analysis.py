# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/monthly-milk-production-pounds-p.csv')
df.head()
df.tail()
df.columns = ['Month','Milk in pounds per cow']
df.head()
# Weird last value at bottom causing issues
df.drop(168,axis=0,inplace=True)
df['Month'] = pd.to_datetime(df['Month'])
df.head()
df.set_index('Month',inplace=True)
df.head()
df.describe().transpose()
# Let's visualize the Data with few methods
df.plot();
timeseries = df['Milk in pounds per cow']
timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(12).std().plot(label='12 Month Rolling Std')
timeseries.plot(figsize = (12,8))
plt.legend();
timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.plot(figsize = (12,8))
plt.legend();
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Milk in pounds per cow'], freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
# Testing for Stationarity
# We can use the Augmented Dickey-Fuller unit root test.
df.head()
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Milk in pounds per cow'])
print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

for value,label in zip(result,labels):
    print(label+' : '+str(value) )
    
if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
# Store in a function!
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
# We have now realized that our data is seasonal (it is also pretty obvious from the plot itself). 
# This means we need to use Seasonal ARIMA on our model. If our data was not seasonal, it means we could use just ARIMA on it.
# First Difference
df['Milk First Difference'] = df['Milk in pounds per cow'] - df['Milk in pounds per cow'].shift(1)
adf_check(df['Milk First Difference'].dropna())
df['Milk First Difference'].plot()
# Second Difference
# Sometimes it would be necessary to do a second difference 
# This is just for show, we didn't need to do a second difference in our case
df['Milk Second Difference'] = df['Milk First Difference'] - df['Milk First Difference'].shift(1)
adf_check(df['Milk Second Difference'].dropna())
df['Milk Second Difference'].plot()
# Seasonal Difference
df['Seasonal Difference'] = df['Milk in pounds per cow'] - df['Milk in pounds per cow'].shift(12)
df['Seasonal Difference'].plot()
# Seasonal Difference by itself was not enough!
adf_check(df['Seasonal Difference'].dropna())
# Seasonal First Difference
df['Seasonal First Difference'] = df['Milk First Difference'] - df['Milk First Difference'].shift(12)
df['Seasonal First Difference'].plot()
adf_check(df['Seasonal First Difference'].dropna())
# Autocorrelation and Partial Autocorrelation Plots
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig_first = plot_acf(df["Milk First Difference"].dropna())
fig_seasonal_first = plot_acf(df["Seasonal First Difference"].dropna())
# Pandas also has this functionality built in, but only for ACF, not PACF.
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Seasonal First Difference'].dropna())
# Partial Autocorrelation
result = plot_pacf(df["Seasonal First Difference"].dropna())
# Final ACF and PACF Plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax2)
# Using the Seasonal ARIMA model
# For non-seasonal data
from statsmodels.tsa.arima_model import ARIMA
# We have seasonal data!
model = sm.tsa.statespace.SARIMAX(df['Milk in pounds per cow'],order=(0,1,0), seasonal_order=(1,1,1,12))
results = model.fit()
print(results.summary())
results.resid.plot()
results.resid.plot(kind='kde')
# Prediction of Future Values
df['forecast'] = results.predict(start = 150, end= 168, dynamic= True)  
df[['Milk in pounds per cow','forecast']].plot(figsize=(12,8))
# Forecasting
# This requires more time periods, so let's create them with pandas onto our original dataframe!
df.tail()
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,24) ]
future_dates
future_dates_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_df = pd.concat([df,future_dates_df])
future_df.head()
future_df.tail()
future_df['forecast'] = results.predict(start = 168, end = 188, dynamic= True)  
future_df[['Milk in pounds per cow', 'forecast']].plot(figsize=(12, 8)) 



