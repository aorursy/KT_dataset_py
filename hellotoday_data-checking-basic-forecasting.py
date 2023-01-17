# Hide all the warnings in this notebook

import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import pathlib



# Read in data

current_path=pathlib.Path().absolute()



df_test =pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

df_train=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

df_submission=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')



print ('test size: ', df_test.shape)

print ('train size: ', df_train.shape)

print ('submission size: ', df_submission.shape)
print(df_test.describe())

print (df_train.describe())

print (df_submission.describe())
# check each numeric value distribution

hist = df_train.hist(bins=5)

hist
# From above results, it's worth to check Fatalities and Confirmed Cases. The data seems in a central spot

df_train[['ConfirmedCases','Fatalities']].plot.kde(bw_method=0.3)
# convert object to date

df_train['Date']=pd.to_datetime(df_train['Date'])

df_train.plot(x='Date', y='ConfirmedCases',kind='line')
# pivot the data to take country as columns for later chart

df_train_pivot=pd.pivot_table(df_train, values='ConfirmedCases', index=['Date'],

                    columns=['Country/Region'], aggfunc=np.sum)

df_train_pivot=df_train_pivot.reset_index()

pd.set_option('display.max_columns', 500)

df_train_pivot.head()
import matplotlib.pyplot as plt



fig, ax = plt.subplots()  

country_set=np.setdiff1d(df_train_pivot.columns,['Country/Region','Date']) 



def countryPlot(countryList):

    for x in countryList:

        ax.plot(df_train_pivot['Date'], df_train_pivot[x], label=x)



countryPlot(country_set)

ax.set_xlabel('Date')  

ax.set_ylabel('Confirmed Cases Count')  

ax.set_title("Confirmed Cases Country BreakDown")  

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=5)  
# Above chart has too many countries and it's difficult to see which country with all the colours

# Next step is to get the top 10 countries with highest confirmed cases



df_train.groupby(['Country/Region'])['ConfirmedCases'].sum().reset_index().sort_values('ConfirmedCases', ascending=False)[:9]
# print out trainning data time range

print('Start Date:',min(df_train['Date']))

print('End Date:', max(df_train['Date']))



# print out test data time range

df_test['Date']=pd.to_datetime(df_test['Date'])

print('Start Date:',min(df_test['Date']))

print('End Date:',max(df_test['Date']))



# as Test dataset doesn't have Confirmed Case number, so we cannot use Test set to monitor the model, so I decided to break the trainning set to 2 parts
# if we take it from global point of view

df_global_train=df_train[df_train['Date']<='2020-03-12'].groupby('Date')['ConfirmedCases'].sum().reset_index()

df_global_test= df_train[df_train['Date']>='2020-03-12'].groupby('Date')['ConfirmedCases'].sum().reset_index()
df_global_train.index=df_global_train['Date']

df_global_test.index=df_global_test['Date']



# from above 2 start dates and end dates, we can see there are overlap between trainning set and test set

# If we take 2020-03-12 as the breaking point

#Plotting data

df_global_train.ConfirmedCases.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

df_global_test.ConfirmedCases.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

plt.show()
# Method 1 - if we are taking Naive Forecast

dd = np.asarray(df_global_train['ConfirmedCases'])

y_hat = df_global_test.copy()

y_hat['naive'] = dd[len(dd) - 1]

plt.figure(figsize=(12, 8))

plt.plot(df_global_train.index, df_global_train['ConfirmedCases'], label='Train')

plt.plot(df_global_test.index, df_global_test['ConfirmedCases'], label='Test')

plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')

plt.legend(loc='best')

plt.title("Naive Forecast")

plt.show()
# From the above chart we already can tell Naive Forecast doesn't fit into this Covid-19

# But we want to measure how poor this model is by using RMS

from sklearn.metrics import mean_squared_error

from math import sqrt

 

rms = sqrt(mean_squared_error(df_global_test['ConfirmedCases'], y_hat['naive']))

print(rms)
# Method 2 - ARIMA

import statsmodels.api as sm

 

y_hat_avg = df_global_test.copy()

fit1 = sm.tsa.statespace.SARIMAX(df_global_train.ConfirmedCases, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()

y_hat_avg['SARIMA'] = fit1.predict(start=min(df_global_test['Date']), end=max(df_global_test['Date']), dynamic=True)

plt.figure(figsize=(16, 8))

plt.plot(df_global_train['ConfirmedCases'], label='Train')

plt.plot(df_global_test['ConfirmedCases'], label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.show()
# From the above chart we already can tell ARIMA doesn't fit into this Covid-19

# But we want to measure how poor this model is by using RMS

from sklearn.metrics import mean_squared_error

from math import sqrt

 

rms = sqrt(mean_squared_error(df_global_test['ConfirmedCases'], y_hat_avg['SARIMA']))

print(rms)