# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

from statsmodels.tsa.stattools import adfuller

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the data

df = pd.read_csv("../input/portland-oregon-average-monthly-.csv")
# A glance on the data 

df.head()
df.columns = ["month", "average_monthly_ridership"]

df.head()
# getting some information about dataset

df.info()
df.dtypes
df.tail()
df = df.iloc[:-1,:]   #removing last row
df['average_monthly_ridership'] = pd.to_numeric(df['average_monthly_ridership'])
df['month'] = pd.to_datetime(df['month'], format = '%Y-%m')
df.dtypes
# Normal line plot so that we can see data variation

# We can observe that average number of riders is increasing most of the time

# We'll later see decomposed analysis of that curve

df.plot.line(x = 'month', y = 'average_monthly_ridership')

plt.show()
rider = df[['average_monthly_ridership']]
log_ridership = np.log(df[['average_monthly_ridership']])
log_ridership.plot.line()
# 1st order differencing

rider_single_diff = (log_ridership.diff()).dropna()  # 1st term will be NAN



#NOTE: diff(diff(X)) is 2nd order differencing 

rider_double_diff = (rider_single_diff.diff()).dropna()  



#seasonal differencing of order 1

rider_single_seasonal_diff = (rider_single_diff.diff(periods=12)).dropna()  # 1st term will be NAN



rider_single_diff.plot.line()

plt.title('1st Order Diff')

plt.show()



rider_double_diff.plot.line()

plt.title('2nd Order Diff')

plt.show()



rider_single_seasonal_diff.plot.line()

plt.title('1st Order Seasonal Diff')

plt.show()
#Perform Dickey–Fuller test:

print('Results of Dickey Fuller Test:')

dftest = adfuller(rider_single_seasonal_diff.average_monthly_ridership, autolag='AIC') #Note: the input should not be a dataframe but a panda series

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
#ACF and PACF plots:

import statsmodels.api as sm

sm.graphics.tsa.plot_acf(rider_single_seasonal_diff.values.squeeze(), lags=40)

plt.title('ACF')

plt.show()



sm.graphics.tsa.plot_pacf(rider_single_seasonal_diff.values.squeeze(), lags=40)

plt.title('PACF')

plt.show()
#for our model we need dates as indexes

df = df.set_index('month')



#doing log transformation on data

df['average_monthly_ridership'] = np.log(df[['average_monthly_ridership']])
# Applying Seasonal ARIMA model to forcast the data 

mod = sm.tsa.SARIMAX(df['average_monthly_ridership'], trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))

results = mod.fit()

print(results.summary())
df['forecast'] = results.predict(start = 102, end= 120, dynamic= True)  

df[['average_monthly_ridership', 'forecast']].plot(figsize=(12, 8))

plt.show()
def forcasting_future_months(df, no_of_months):

    df_predict = df.reset_index()

    mon = df_predict['month']

    mon = mon + pd.DateOffset(months = no_of_months)

    future_dates = mon[-no_of_months -1:]

    df_predict = df_predict.set_index('month')

    future = pd.DataFrame(index=future_dates, columns= df_predict.columns)

    df_predict = pd.concat([df_predict, future])

    df_predict['forecast'] = results.predict(start = 114, end = 125, dynamic= True)  

    df_predict[['average_monthly_ridership', 'forecast']].iloc[-no_of_months - 12:].plot(figsize=(12, 8))

    plt.show()

    return df_predict[-no_of_months:]
predicted = forcasting_future_months(df,10)
df = df.apply(np.exp)

forecast = predicted.apply(np.exp)

final = df.append(forecast)

final[['average_monthly_ridership', 'forecast']].plot(figsize=(12, 8))