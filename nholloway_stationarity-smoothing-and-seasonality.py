import os

import math

import pandas as pd

import numpy as np

from collections import defaultdict

from scipy.stats import boxcox

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

from statsmodels.tsa.filters.hp_filter import hpfilter

from matplotlib import pyplot as plt



plt.style.use('fivethirtyeight') 

plt.rcParams['xtick.labelsize'] = 20

plt.rcParams['ytick.labelsize'] = 20



%matplotlib inline 
path = '../input/csvs_per_year/csvs_per_year'

files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

df = pd.concat((pd.read_csv(file) for file in files), sort=False)

df = df.groupby(['date']).agg('mean')

df.index = pd.DatetimeIndex(data= df.index)
'''

['CH4', 'MXY', 'NO', 'NOx', 'OXY', 'PM25', 'PXY'] 

Have been removed because they were missing large potions of data

'''

col_list = ['BEN', 'CO', 'EBE', 'NMHC', 'NO_2', 'O_3', 'PM10', 'SO_2', 'TCH', 'TOL']

monthly_df = df.resample('M').mean()

daily_df = df.resample('D').mean()



plt_monthly = monthly_df[col_list]

plt_monthly.plot(figsize=(15, 10))

plt.title('Madrid Pollutants from 2001-2019', fontsize=25)

plt.legend(loc='upper left')

plt.show()
def adf_test(timeseries):

    print ('Results of Dickey-Fuller Test:')

    print('Null Hypothesis: Unit Root Present')

    print('Test Statistic < Critical Value => Reject Null')

    print('P-Value =< Alpha(.05) => Reject Null\n')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput[f'Critical Value {key}'] = value

    print (dfoutput, '\n')



def kpss_test(timeseries, regression='c'):

    # Whether stationary around constant 'c' or trend 'ct

    print ('Results of KPSS Test:')

    print('Null Hypothesis: Data is Stationary/Trend Stationary')

    print('Test Statistic > Critical Value => Reject Null')

    print('P-Value =< Alpha(.05) => Reject Null\n')

    kpsstest = kpss(timeseries, regression=regression)

    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])

    for key,value in kpsstest[3].items():

        kpss_output[f'Critical Value {key}'] = value

    print (kpss_output, '\n')
sta_so2_df = pd.DataFrame(monthly_df.SO_2)

sta_so2_df['boxcox_SO_2'], lamda = boxcox(sta_so2_df.SO_2)

sta_so2_df.plot(figsize=(13, 5), title='SO_2 Original and Box-Cox Transformed')

plt.show()

print('Box-Cox Lambda Value: ', lamda)

sta_so2_df.plot(figsize=(10, 5), title= 'SO_2 Distribution Before and After Box-Cox', kind='KDE')

plt.show()

sta_pm10_df = pd.DataFrame(monthly_df.PM10)

sta_pm10_df['boxcox_PM10'], pm_lamda = boxcox(sta_pm10_df.PM10)

sta_pm10_df.plot(figsize=(13, 5), title='PM10 Original and Box-Cox Transformed')

plt.show()

print('Box-Cox Lambda Value: ', pm_lamda)

sta_pm10_df.plot(figsize=(10, 5), title= 'PM10 Distribution Before and After Box-Cox', kind='KDE')

plt.show()
sta_so2_df['SO_diff1'] = sta_so2_df['boxcox_SO_2'].diff()

sta_so2_df.SO_diff1.dropna(inplace=True)

adf_test(sta_so2_df.SO_diff1)

kpss_test(sta_so2_df.SO_diff1)

sta_so2_df[['SO_diff1', 'boxcox_SO_2']].plot(figsize=(13, 10), title='SO_2 Before and After Differencing')

sta_so2_df[['SO_diff1', 'SO_2']].plot(figsize=(13, 10), title='SO_2 Original and Stationary')
sta_pm10_df['PM10_diff1'] = sta_pm10_df['boxcox_PM10'].diff()

sta_pm10_df.PM10_diff1.dropna(inplace=True)

adf_test(sta_pm10_df.PM10_diff1)

kpss_test(sta_pm10_df.PM10_diff1)

sta_pm10_df[['PM10_diff1', 'boxcox_PM10']].plot(figsize=(13, 10), title='PM10 Before and After Differencing')

sta_pm10_df[['PM10_diff1', 'PM10']].plot(figsize=(13, 10), title='PM10 Original and Stationary')
acf1 = plot_acf(sta_so2_df['SO_2'])

acf2 = plot_acf(sta_so2_df['SO_diff1'])

acf1.suptitle('Autocorrelation SO_2 Original and Stationary', fontsize=25)

acf1.set_figheight(10)

acf1.set_figwidth(15)

acf2.set_figheight(10)

acf2.set_figwidth(15)

plt.show()

pacf1 = plot_pacf(sta_so2_df['SO_2'], lags=50)

pacf2 = plot_pacf(sta_so2_df['SO_diff1'], lags=50)

pacf1.suptitle('Partial Autocorrelation SO_2 Original and Stationary', fontsize=25)

pacf1.set_figheight(10)

pacf1.set_figwidth(15)

pacf2.set_figheight(10)

pacf2.set_figwidth(15)

plt.show()
SMA_df= pd.DataFrame(monthly_df.NO_2)

for n in range(1, 6):

    label = 'window_'+str(n*3)

    SMA_df[label] = SMA_df['NO_2'].rolling(window=n*3).mean()



SMA_df.plot(figsize=(10, 5), title='NO_2 Moving Averages')

plt.show()



error = {}

for col in SMA_df.columns[1:]:

    series = SMA_df[['NO_2', col]]

    series = series.dropna(axis=0)

    error[col] = mean_squared_error(series['NO_2'], series[col]) 

    print(f'Moving Average {col} MSE: ', round(error[col], 2))



metadata = {'mean':[], 'variance':[], 'skew':[], 'kurtosis':[]}

for col in SMA_df:

    metadata['mean'].append(SMA_df[col].mean())

    metadata['variance'].append(SMA_df[col].var())

    metadata['skew'].append(SMA_df[col].skew())

    metadata['kurtosis'].append(SMA_df[col].kurt())

    

plt.figure(figsize=(10,10))

plt.suptitle('Changes in Series Distribution as Window Size Increases')

plt.subplot(221)

plt.title('NO2 Mean')

plt.plot(metadata['mean'])

plt.subplot(222)

plt.title('NO2 Variance')

plt.plot(metadata['variance'])

plt.subplot(223)

plt.title('NO2 Skewness')

plt.plot(metadata['skew'])

plt.subplot(224)

plt.title('NO2 Kurtosis')

plt.plot(metadata['kurtosis'])

plt.show()



plt.figure(figsize=(10, 15))

plt.subplot(311)

plt.title('Original Time Series Distribution')

SMA_df.NO_2.plot(kind='kde')

plt.subplot(312)

plt.title('Moving Average 6 Time Series Distribution')

SMA_df['window_6'].plot(kind='kde')

plt.subplot(313)

plt.title('Moving Average 15 Time Series Distribution')

SMA_df['window_15'].plot(kind='kde')

plt.show()
smoothing_df = pd.DataFrame(monthly_df.O_3)

smoothing_df.plot(figsize=(10, 5), title='Original Monthly O_3 Time Series')

plt.show()

es_simple = SimpleExpSmoothing(smoothing_df.O_3).fit()

es_double = ExponentialSmoothing(smoothing_df.O_3, trend= 'add', damped= True, seasonal= None, seasonal_periods= None).fit()

es_triple = ExponentialSmoothing(smoothing_df.O_3, trend= 'add', damped= True, seasonal= 'mul', seasonal_periods= 12).fit()



plt.figure(figsize=(10, 5))

es_simple.fittedvalues.plot(color=[(230/255,159/255,0)], legend=True, title='Exponentially Smoothed and Forecast O_3')

es_simple.forecast(36).plot(color=[(230/255,159/255,0)], style='-.')

es_double.fittedvalues.plot(color=[(0,158/255,115/255)])

es_double.forecast(36).plot(color=[(0,158/255,115/255)], style='-.')

es_triple.fittedvalues.plot(color=[(0,114/255,178/255)])

es_triple.forecast(36).plot(color=[(0,114/255,178/255)], style='-.')



plt.legend(['ses', 'ses forecast', 'double exp', 'double exp forecast', 

            'triple exp', 'triple exp forecast'])

plt.show()



smoothing_df['simple_es'] = es_simple.fittedvalues

smoothing_df['double_es'] = es_double.fittedvalues

smoothing_df['triple_es'] = es_triple.fittedvalues

print('Simple Exponential Smoothing MSE: ', round(es_simple.sse/len(es_simple.fittedvalues), 2))

print('Double Exponential Smoothing MSE: ', round(es_double.sse/len(es_double.fittedvalues), 2))

print('Triple Exponential Smoothing MSE: ', round(es_triple.sse/len(es_triple.fittedvalues), 2))



plt.figure(figsize=(10, 20))

plt.subplot(411)

plt.title('Original Monthly O_3 Distribution')

smoothing_df.O_3.plot(kind='kde')

plt.subplot(412)

plt.title('Simple Exponential Smoothing Distribution')

smoothing_df['simple_es'].plot(kind='kde')

plt.subplot(413)

plt.title('Double Exponential Smoothing Distribution')

smoothing_df['double_es'].plot(kind='kde')

plt.subplot(414)

plt.title('Triple Exponential Smoothing')

smoothing_df['triple_es'].plot(kind='kde')

plt.show()
decomposition_df = pd.DataFrame(monthly_df.O_3)

seasonal_a = seasonal_decompose(decomposition_df, model='additive')

seasonal_m = seasonal_decompose(decomposition_df, model='multiplicative')

fig_1 = seasonal_a.plot()

fig_2 = seasonal_m.plot()

fig_1.suptitle('Additive Seasonal Decomposition', fontsize=25)

fig_1.set_figheight(10)

fig_1.set_figwidth(20)

fig_2.suptitle('Multiplicative Seasonal Decomposition', fontsize=25)

fig_2.set_figheight(10)

fig_2.set_figwidth(20)

plt.show()

filter_df = pd.DataFrame(monthly_df.O_3)

O3_cycle, O3_trend = hpfilter(filter_df, lamb=129600)

filter_df['cycle'] = O3_cycle

filter_df['trend'] = O3_trend



filter_df.plot(figsize=(10, 5), title='O_3 Pollutant Plot of Cycle and Trend')