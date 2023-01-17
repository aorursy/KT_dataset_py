

import numpy as np

import pandas as pd

import gc

import os

import random

import copy

import matplotlib.pyplot as plt

import pandas

import statsmodels

from statsmodels.tsa.stattools import adfuller
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df=pd.read_csv('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/amzn.us.txt',parse_dates=['Date'],index_col=['Date'],date_parser=dateparse)

df=df.drop(columns=['Open','High','Low','Volume','OpenInt'])

df.plot(figsize=(15,6))

plt.show()
adftest=adfuller(df['Close'])

print('ADF Statistic: %f' % adftest[0])

print('ADF p-value: %f' % adftest[1])
df_log=pd.DataFrame(np.log(df['Close']),index=df.index)

plt.figure(figsize=(15,6))

plt.plot(df_log)

plt.title('Log Close')

plt.show()
adftest=adfuller(df_log['Close'])

print('ADF Statistic: %f' % adftest[0])

print('ADF p-value: %f' % adftest[1])
df['Returns']=df['Close']-df['Close'].shift(1)

df_log['Log_Returns']=df_log['Close']-df_log['Close'].shift(1)

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

plt.plot(df['Returns'])

plt.title('Returns')

plt.subplot(1,2,2)

plt.plot(df_log['Log_Returns'],'g-')

plt.title('Log Returns')

plt.show()
adftest_ret=adfuller(df['Returns'].dropna())

print('Returns ADF Statistic: %f' % adftest_ret[0])

print('Returns ADF p-value: %f' % adftest_ret[1])

adftest_lret=adfuller(df_log['Log_Returns'].dropna())

print('Log-Returns ADF Statistic: %f' % adftest_lret[0])

print('Log-Returns ADF p-value: %f' % adftest_lret[1])
def getWeights(d,lags):

    # return the weights from the series expansion of the differencing operator

    # for real orders d and up to lags coefficients

    w=[1]

    for k in range(1,lags):

        w.append(-w[-1]*((d-k+1))/k)

    w=np.array(w).reshape(-1,1) 

    return w

def plotWeights(dRange, lags, numberPlots):

    weights=pd.DataFrame(np.zeros((lags, numberPlots)))

    interval=np.linspace(dRange[0],dRange[1],numberPlots)

    for i, diff_order in enumerate(interval):

        weights[i]=getWeights(diff_order,lags)

    weights.columns = [round(x,2) for x in interval]

    fig=weights.plot(figsize=(15,6))

    plt.legend(title='Order of differencing')

    plt.title('Lag coefficients for various orders of differencing')

    plt.xlabel('lag coefficients')

    #plt.grid(False)

    plt.show()

def ts_differencing(series, order, lag_cutoff):

    # return the time series resulting from (fractional) differencing

    # for real orders order up to lag_cutoff coefficients

    

    weights=getWeights(order, lag_cutoff)

    res=0

    for k in range(lag_cutoff):

        res += weights[k]*series.shift(k).fillna(0)

    return res[lag_cutoff:] 
plotWeights([0.1,0.9],20,5)
differences=[0.5,0.9]

fig, axs = plt.subplots(len(differences),2,figsize=(15,6))

for i in range(0,len(differences)):

    axs[i,0].plot(ts_differencing(df['Close'],differences[i],20))

    axs[i,0].set_title('Original series with d='+str(differences[i]))

    axs[i,1].plot(ts_differencing(df_log['Close'],differences[i],20),'g-')

    axs[i,1].set_title('Logarithmic series with d='+str(differences[i]))

    plt.subplots_adjust(bottom=0.01) #increasing space between plots for aestethics
def cutoff_find(order,cutoff,start_lags): #order is our dearest d, cutoff is 1e-5 for us, and start lags is an initial amount of lags in which the loop will start, this can be set to high values in order to speed up the algo

    val=np.inf

    lags=start_lags

    while abs(val)>cutoff:

        w=getWeights(order, lags)

        val=w[len(w)-1]

        lags+=1

    return lags
def ts_differencing_tau(series, order, tau):

    # return the time series resulting from (fractional) differencing

    lag_cutoff=(cutoff_find(order,tau,1)) #finding lag cutoff with tau

    weights=getWeights(order, lag_cutoff)

    res=0

    for k in range(lag_cutoff):

        res += weights[k]*series.shift(k).fillna(0)

    return res[lag_cutoff:] 
#this part takes about 20 minutes to compute

possible_d=np.divide(range(1,100),100)

tau=1e-4

original_adf_stat_holder=[None]*len(possible_d)

log_adf_stat_holder=[None]*len(possible_d)



for i in range(len(possible_d)):

    original_adf_stat_holder[i]=adfuller(ts_differencing_tau(df['Close'],possible_d[i],tau))[1]

    log_adf_stat_holder[i]=adfuller(ts_differencing_tau(df_log['Close'],possible_d[i],tau))[1]
#now the plots of the ADF p-values

fig, axs = plt.subplots(1,2,figsize=(15,6))

axs[0].plot(possible_d,original_adf_stat_holder)

axs[0].axhline(y=0.01,color='r')

axs[0].set_title('ADF P-value by differencing order in the original series')

axs[1].plot(possible_d,log_adf_stat_holder)

axs[1].axhline(y=0.01,color='r')

axs[1].set_title('ADF P-value by differencing order in the logarithmic series')