import numpy as np

import csv

import matplotlib.pyplot as plt

import pandas as pd

#get_ipython().magic('matplotlib')

import datetime

df=pd.read_csv('../input/Facebook_Groups_Rare_Diseases_V4.csv',';',encoding='latin1',parse_dates=['post_published'], infer_datetime_format=True)

to_del = ['encounter_id', 'patient_nbr','medical_specialty','payer_code']

print(df.columns)
for i in range(0, len(df)):

    aString=df.loc[i,'post_published']

    if ('T' in aString):

        aString.replace('T',' ')

        df.loc[i,'post_published']=aString[:-5]

    else:

        df=df.drop(i)
print(df['post_published'].head(10))

df['post_published'] = pd.to_datetime(df['post_published'],format='%Y-%m-%d %H:%M:%S')

df = df.set_index(df['post_published'])
print(df.index)

print (df.dtypes)
ts = df['engagement_fb'] 

print(ts.head(10))
df2=df.sort_index()
ts = df2['engagement_fb'] 

print(ts.head(10))
plt.plot(ts)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = pd.rolling_mean(timeseries, window=12)

    rolstd = pd.rolling_std(timeseries, window=12)



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)

test_stationarity(ts)
#Estimating and eliminating trend

ts_log = np.log(ts)

plt.plot(ts_log)

moving_avg = pd.rolling_mean(ts_log,12)

plt.plot(ts_log)

plt.plot(moving_avg, color='red')
#Moving average

ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_log_moving_avg_diff)
#Exponentially weighted moving average

expwighted_avg = pd.ewma(ts_log, halflife=12)

plt.plot(ts_log)

plt.plot(expwighted_avg, color='red')
ts_log_diff = ts_log - ts_log.shift()

plt.plot(ts_log_diff)
#Auto-Regressive Integrated Moving Averages

threshold_for_bug = 0.00000001 # could be any value, ex numpy.min

ts_log[ts_log < threshold_for_bug] = threshold_for_bug

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
#MA model

model = ARIMA(ts_log, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
#Combined model

model = ARIMA(ts_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))