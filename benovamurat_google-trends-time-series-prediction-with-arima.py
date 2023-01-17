#!pip install pytrends
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

plt.style.use('fivethirtyeight')
np.random.seed(777)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
pytrend = TrendReq(hl='tr-TR')

start_date = '2017-01-01'
end_date = '2019-12-31'
date_range = start_date + ' ' + end_date

st = ['migros']

pytrend.build_payload(kw_list=st, timeframe = date_range)

trends = pytrend.interest_over_time()
df = trends.drop(columns='isPartial')
df.reset_index(inplace=True)
df.head()
df.info()
df.columns=['Date', 'Trend']
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.head()
df.describe().transpose()
df.plot()
timeseries = df['Trend']
timeseries.rolling(52).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(52).std().plot(label='12 Month Rolling Std')
timeseries.plot()
plt.legend()
timeseries.rolling(52).mean().plot(label='12 Month Rolling Mean')
timeseries.plot()
plt.legend()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Trend'], period=52) 
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(15, 8)
df.head()
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Trend'])
print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

for value,label in zip(result,labels):
    print(label+' : '+str(value) )
    
if result[1] <= 0.05:
    print('Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary.')
else:
    print('Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.')
# Store in a function for later use!
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
        print('Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary.')
    else:
        print('Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.')
df['Trend First Difference'] = df['Trend'] - df['Trend'].shift(1)
adf_check(df['Trend'].dropna())
df['Trend First Difference'].plot()
# Sometimes it would be necessary to do a second difference 
# This is just for show, we didn't need to do a second difference in our case
df['Trend Second Difference'] = df['Trend First Difference'] - df['Trend First Difference'].shift(1)
adf_check(df['Trend Second Difference'].dropna())
df['Trend Second Difference'].plot()
df['Seasonal Difference'] = df['Trend'] - df['Trend'].shift(52)
df['Seasonal Difference'].plot()
adf_check(df['Seasonal Difference'].dropna())
df['Seasonal First Difference'] = df['Trend First Difference'] - df['Trend First Difference'].shift(52)
df['Seasonal First Difference'].plot()
adf_check(df['Seasonal First Difference'].dropna())
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# Duplicate plots
# Check out: https://stackoverflow.com/questions/21788593/statsmodels-duplicate-charts
# https://github.com/statsmodels/statsmodels/issues/1265
fig_first = plot_acf(df['Trend First Difference'].dropna())
fig_seasonal_first = plot_acf(df['Seasonal First Difference'].dropna())
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Seasonal First Difference'].dropna())
result = plot_pacf(df['Seasonal First Difference'].dropna())
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[53:], lags=52, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[53:], lags=52, ax=ax2)
from statsmodels.tsa.arima_model import ARIMA
help(ARIMA)
model = sm.tsa.statespace.SARIMAX(df['Trend'], order=(0,1,0), seasonal_order=(1,1,1,52))
results = model.fit()
print(results.summary())
results.resid.plot()
results.resid.plot(kind='kde')
df['forecast'] = results.predict(start = 52, end= 400, dynamic= True)  
df[['Trend','forecast']].plot(figsize=(12,8))
df.tail()
# https://pandas.pydata.org/pandas-docs/stable/timeseries.html
# Alternatives 
# pd.date_range(df.index[-1],periods=12,freq='M')
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset(weeks=x) for x in range(0,52) ]
#future_dates
future_dates_df = pd.DataFrame(index=future_dates[1:], columns=df.columns) 
future_df = pd.concat([df, future_dates_df])
future_df.head()
future_df.tail()
future_df['forecast'] = results.predict(start=157, end=312, dynamic=True)  
future_df[['Trend', 'forecast']].plot(figsize=(12, 8))
future_df[['Trend', 'forecast']]