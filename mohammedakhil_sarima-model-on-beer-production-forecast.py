import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#Lets read in the data
df = pd.read_csv("../input/monthly-beer-production/datasets_56102_107707_monthly-beer-production-in-austr.csv")
df.head()
#lets look at the brief summary of the Dataframe
df.info()

df['Month']=pd.to_datetime(df['Month'])
df.dtypes
# Lets set the month column as the index for our Dataframe
df.set_index('Month',inplace=True)
# Basic plot to get the general idea of the trends in data
df.plot()
df.iloc[:60].plot()
from statsmodels.tsa.stattools import adfuller
test=adfuller(df['Monthly beer production'])
print (test)
annual_difference = df['Monthly beer production'] - df['Monthly beer production'].shift(12)
adfuller(annual_difference.dropna())
df['Annual Difference']=annual_difference
(df['Annual Difference'].iloc[12:48]).plot()
df.head()
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Annual Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Annual Difference'].iloc[13:],lags=40,ax=ax2)
from statsmodels.tsa.arima_model import ARIMA
x_test=df['Monthly beer production'].iloc[:400]
# Create Model with first 400 values and use it to predict the remaining values inorder to check accuracy
model=ARIMA(x_test,order=(1,1,1))
model_fit=model.fit()
df['ARIMA forecast']=model_fit.predict(start=400,end=470,dynamic=True)
df[['Monthly beer production','ARIMA forecast']].plot(figsize=(12,8))
model=sm.tsa.statespace.SARIMAX(x_test,order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['SARIMA forecast']=results.predict(start=400,end=470,dynamic=True)
df[['Monthly beer production','SARIMA forecast']].plot(figsize=(12,8))
df[['Monthly beer production','SARIMA forecast']].iloc[301:470].plot(figsize=(12,8))
residuals=df['Monthly beer production']-df['SARIMA forecast']
residuals.dropna(inplace=True)
print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/df['Monthly beer production'].iloc[400:471])),4))
print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))
residuals.plot()
from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_df=pd.concat([df,future_datest_df])
future_df['futureforecast'] = results.predict(start = 474, end = 500, dynamic= True)  
future_df[['Monthly beer production', 'futureforecast']].plot(figsize=(12, 8))
future_df[['Monthly beer production', 'futureforecast']].iloc[400:].plot(figsize=(8, 8))