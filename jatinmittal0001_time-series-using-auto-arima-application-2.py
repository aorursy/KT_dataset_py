import warnings

warnings.filterwarnings('ignore')

# To  collect garbage (delete files)

import gc

# To save dataset as pcikle file for future use

import pickle



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# for basic math operations like sqrt

import math





from sklearn.svm import SVR

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import plot_importance

from sklearn.linear_model import LinearRegression

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



            

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error



import os

print(os.listdir("../input/rapido"))
input_data = pd.read_csv("../input/rapido-rides/ct_rr.csv")

input_data.shape
print("Data size before removing: ",input_data.shape)



# Check duplicated rows in train set

df = input_data[input_data.duplicated()]  # checks duplicate rows considering all columns

print("Number of duplicate observations: ", len(df))

del df

gc.collect();



#Dropping duplicates and keeping first occurence only

input_data.drop_duplicates(keep = 'first', inplace = True)



print("Data size after removing: ",input_data.shape)
input_data['ts'] = pd.to_datetime(input_data['ts'], format='%Y%m%')

input_data['ts'] = input_data['ts'].dt.date   # converting column Month to desired Year and month format and having datetime datatype
input_data['ts'] = pd.to_datetime(input_data['ts'])
input_data.head()
ab = input_data.groupby(['ts']).number.count()

ab = ab.reset_index()

ab.head()
ab.tail()
ab = ab[ab['ts']<'2019-04-01']

print(ab.shape)

ab['number'].plot.line()
ab = ab.set_index('ts').resample('D').ffill()

ab = ab.reset_index()
ab['number'].iloc[10:40].plot.line()
rides = ab[['number']]
log_ridership = np.log(rides)

log_ridership.plot.line()
# 1st order differencing

rider_single_diff = (log_ridership.diff()).dropna()  # 1st term will be NAN



#NOTE: diff(diff(X)) is 2nd order differencing 

rider_double_diff = (rider_single_diff.diff()).dropna()  



#seasonal differencing of order 1

rider_single_seasonal_diff = (rider_single_diff.diff(periods=7)).dropna()  # 1st term will be NAN

rider_double_seasonal_diff = (rider_double_diff.diff(periods=7)).dropna()  # 1st term will be NAN



rider_single_diff.plot.line()

plt.title('1st Order Diff')

plt.show()



rider_double_diff.plot.line()

plt.title('2nd Order Diff')

plt.show()



rider_single_seasonal_diff.plot.line()

plt.title('1st Order Seasonal Diff with seasonality of 7 days')

plt.show()
#Perform Dickey–Fuller test:

print('Results of Dickey Fuller Test:')

dftest = adfuller(rider_double_seasonal_diff.number, autolag='AIC') #Note: the input should not be a dataframe but a panda series

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
#ACF and PACF plots:



# Here d=1, D=1 and log transformed

import statsmodels.api as sm

sm.graphics.tsa.plot_acf(rider_single_seasonal_diff.values.squeeze(), lags=40)

plt.title('ACF')

plt.show()



sm.graphics.tsa.plot_pacf(rider_single_seasonal_diff.values.squeeze(), lags=40)

plt.title('PACF')

plt.show()
#for our model we need dates as indexes

ab = ab.set_index('ts')



#doing log transformation on main data

ab['number'] = np.log(ab[['number']])
!pip install pmdarima
from pmdarima.arima import auto_arima

stepwise_model = auto_arima(ab['number'], start_p=1, start_q=1,

                           max_p=2, max_q=1, m=7,

                           start_P=1,max_P=2, seasonal=True,

                           d=1, D=1, max_d = 2, max_D=2,trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)

print(stepwise_model.aic())
# Applying Seasonal ARIMA model to forcast the data 

mod = sm.tsa.SARIMAX(ab['number'], trend='n', order=(0,1,1), seasonal_order=(1,1,1,7))  #also play with "trend" argument

results = mod.fit()

print(results.summary())



results.plot_diagnostics(figsize=(15,12))

plt.show()
ab['number'].shape
ab['forecast'] = results.predict(start = 345, end= 359, dynamic= True)  

ab[['number', 'forecast']].iloc[-60:].plot(figsize=(12, 8))

plt.show()
def forcasting_future_months(df, no_of_periods):

    df_predict = df.reset_index()

    mon = df_predict['ts']

    mon = mon + pd.DateOffset(days = no_of_periods)

    future_dates = mon[-no_of_periods -1:]

    df_predict = df_predict.set_index('ts')

    future = pd.DataFrame(index=future_dates, columns= df_predict.columns)

    df_predict = pd.concat([df_predict, future])

    df_predict['forecast'] = results.predict(start = 359, end = 390, dynamic= True)  

    df_predict[['number', 'forecast']].iloc[-60:].plot(figsize=(12, 8))

    plt.show()

    return df_predict[-no_of_periods:]
predicted = forcasting_future_months(ab,30)  #insert dataframe name and number of period for which to forecast
ab = ab.apply(np.exp)

forecast = predicted.apply(np.exp)

final = ab.append(forecast)

final[['number', 'forecast']].plot(figsize=(12, 8))
final.tail(4)
# Sum of num of rides in April

april_rides = final.loc[final.index>'2019-03-31'].forecast.sum()

print(round(april_rides))
per_month_rides =ab.groupby([(ab.index.year),(ab.index.month)]).number.sum()
rides_till_march = pd.Series(per_month_rides.values)

rides_till_april = rides_till_march.append(pd.Series(april_rides))

rides_till_april = rides_till_april.reset_index(drop = True) 
rides_till_april
plt.scatter(rides_till_april.index, rides_till_april)

plt.show()