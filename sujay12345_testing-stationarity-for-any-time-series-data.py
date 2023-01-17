import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pylab import rcParams



import seaborn as sns

worldometers=pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

worldometersnew=worldometers[['Country/Region','TotalCases','TotalDeaths','TotalRecovered','Serious,Critical']].copy()

worldometersnew.head()

worldometersnew.iloc[0:6].style.background_gradient(cmap='Reds')
sns.set(style='whitegrid')



from pylab import rcParams

rcParams['figure.figsize'] = 5, 7



worldometersnew.iloc[0:4].plot(x='Country/Region',y='TotalCases',kind='bar',color='Red') # first five rows of dataframe

plt.legend(['Total Cases In Top 4 Countries'], loc='upper right')




worldometersnew.iloc[5:10].plot(x='Country/Region',y='TotalCases',kind='barh',color='black') # first five rows of dataframe



rcParams['figure.figsize'] = 4, 3







rcParams['figure.figsize'] = 20, 5

worldometersnew.iloc[10:50].plot(x='Country/Region',y='TotalCases',kind='bar',color='Black') # first five rows of dataframe



plt.legend(['Rest Other COuntries Total Cases'], loc='upper right')
timeseries=pd.read_csv('../input/corona-virus-report/day_wise.csv')

timeseriesnew=timeseries[['Date','New recovered']].copy()



timeseriesnew['Date']=pd.to_datetime(timeseriesnew['Date']) 



timeseriesnew.head()

timeseriesnew['month'] = timeseriesnew['Date'].apply(lambda x: x.month)

timeseriesnew.set_index('Date', inplace= True)

timeseriesnew=timeseriesnew.fillna(method='ffill')



timeseriesyear=timeseriesnew.copy()

tsf=timeseriesnew.loc['2020-01-16':'2020-07-15'].copy()

tsf.drop('month', axis=1,inplace=True)

plt.figure(figsize=(25,8))

plt.plot(tsf)

plt.title('Daily Recovered Cases')

plt.xlabel('Daily Recovering Cases From January 2020 To July 2020 : Each Column Grid Represents Each Month')

plt.ylabel('Number Of Cases : Time Series')

plt.show()

timeseries=pd.read_csv('../input/corona-virus-report/day_wise.csv')

timeseriesnew=timeseries[['Date','New deaths']].copy()









timeseriesnew['Date']=pd.to_datetime(timeseriesnew['Date']) 



timeseriesnew.head()

timeseriesnew['month'] = timeseriesnew['Date'].apply(lambda x: x.month)

timeseriesnew.set_index('Date', inplace= True)

timeseriesnew=timeseriesnew.fillna(method='ffill')



timeseriesyear=timeseriesnew.copy()

tsf=timeseriesnew.loc['2020-01-16':'2020-07-15'].copy()

tsf.drop('month', axis=1,inplace=True)

plt.figure(figsize=(25,8))

plt.plot(tsf)

plt.title('Daily New Deaths')

plt.xlabel('Daily Deaths From January 2020 To July 2020 : Each Column Grid Represents Each Month')

plt.ylabel('Number Of Cases : Time Series')

plt.show()
sns.set_style("whitegrid")

timeseries=pd.read_csv('../input/corona-virus-report/day_wise.csv')

timeseriesnew=timeseries[['Date','New cases']].copy()

timeseriesnew['Date']=pd.to_datetime(timeseriesnew['Date']) 



timeseriesnew.head()

timeseriesnew['month'] = timeseriesnew['Date'].apply(lambda x: x.month)

timeseriesnew.set_index('Date', inplace= True)

timeseriesnew=timeseriesnew.fillna(method='ffill')



timeseriesnew.sample(10)
timeseriesyear=timeseriesnew.copy()

tsf=timeseriesnew.loc['2020-05-16':'2020-07-15'].copy()

tsf.drop('month', axis=1,inplace=True)

plt.figure(figsize=(25,8))

plt.plot(tsf)

plt.title('Daily New Cases')

plt.xlabel('Daily Cases From January 2020 To July 2020 : Each Column Grid Represents Each Month')

plt.ylabel('Number Of Cases : Time Series')

plt.show()
#HO : It is not stationary

#H1 : It is stationary

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of :')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    if dfoutput[1]<=0.05:

        print("Strong Rejection against null hypothesis(HO). Reject Null Hypothesis. The Data is stationary. ")

    else:

        print("Weak evidence against null hypothesis.Time series has a unit root. Therefore it is not stationary.")
plt.figure(figsize=(40,8))

test_stationarity(tsf)

plt.figure(figsize=(40,8))

ts_log = np.log(tsf)

plt.plot(ts_log)
plt.figure(figsize=(40,8))

plt.figure(figsize=(40,8))

moving_avg = ts_log.rolling(12).mean()

plt.plot(ts_log)

plt.plot(moving_avg, color='red')
plt.figure(figsize=(40,8))

ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diffwithoutnull=ts_log_moving_avg_diff.replace([np.inf, -np.inf], np.nan)

ts_log_moving_avg_diffwithoutnull.dropna(0,inplace=True)

test_stationarity(ts_log_moving_avg_diffwithoutnull)
plt.figure(figsize=(40,8))

ts_log_diff = ts_log - ts_log.shift()

ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)

lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')

#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
tss=ts_log.replace([np.inf, -np.inf], np.nan)

tss.dropna(0,inplace=True)

tss.head()
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_AR.fittedvalues, color='red')
model = ARIMA(ts_log, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_MA.fittedvalues, color='red')
model = ARIMA(ts_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')