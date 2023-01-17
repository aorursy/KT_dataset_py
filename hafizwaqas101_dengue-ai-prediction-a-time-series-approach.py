

import pandas as pd

import numpy as np

from statsmodels.tsa.api import VAR

from statsmodels.tsa.stattools import adfuller

from statsmodels.tools.eval_measures import rmse, aic

from statsmodels.tsa.stattools import grangercausalitytests

from statsmodels.stats.stattools import durbin_watson

from statsmodels.tsa.stattools import acf

from math import sqrt

from warnings import catch_warnings

from warnings import filterwarnings

from sklearn.metrics import mean_squared_error

from pandas import read_csv

import io

import matplotlib.pyplot as plt

from numpy import array

from datetime import datetime





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df1 = pd.read_csv('/kaggle/input/dengueai/dengue_features_train.csv', index_col=['city','year','weekofyear'])

df2 = pd.read_csv('/kaggle/input/dengueai/dengue_labels_train.csv', index_col=['city','year','weekofyear'])
df = df1.join(df2)
df.head()
dfsj = df.iloc[ :936]

dfiq = df.iloc[936: ]
dfsj.set_index('week_start_date', inplace=True)

dfiq.set_index('week_start_date', inplace=True)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

fig.suptitle('Cases in San Joun & Iquitos')

fig.set_size_inches(16,5)

ax = dfsj['total_cases'].plot(ax=ax1);

ax1.set_title('San Joun cases')

ax = dfiq['total_cases'].plot(ax=ax2);

ax2.set_title('Iquitos cases')
dfsj.head()
dfiq.head()
fig, axes = plt.subplots(nrows=21, ncols=1, dpi=120, figsize=(12,30))

for i, ax in enumerate(axes.flatten()):

    data = dfsj[dfsj.columns[i]]

    ax.plot(data, color='red', linewidth=1)

    # Decorations

    ax.set_title(dfsj.columns[i])

    ax.xaxis.set_ticks_position('none')

    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)

    ax.tick_params(labelsize=6)



plt.tight_layout();
dfsj.isnull().sum()
dfsj = dfsj.interpolate()

dfsj.isnull().sum()
fig, axes = plt.subplots(nrows=21, ncols=1, dpi=120, figsize=(12,30))

for i, ax in enumerate(axes.flatten()):

    data = dfiq[dfiq.columns[i]]

    ax.plot(data, color='red', linewidth=1)

    # Decorations

    ax.set_title(dfsj.columns[i])

    ax.xaxis.set_ticks_position('none')

    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)

    ax.tick_params(labelsize=6)



plt.tight_layout();
dfiq.isnull().sum()
dfiq = dfiq.interpolate()

dfiq.isnull().sum()
sj_correlations = dfsj.corr()
import seaborn as sns

fig, ax = plt.subplots(figsize=(20,10))

sns.set(style="ticks", palette="colorblind")

(sj_correlations

     .total_cases

     .drop('total_cases') # don't compare with myself

     .sort_values(ascending=False)

     .plot

     .barh()

)
dfsj.shape
dfsj.columns
iq_correlations = dfiq.corr()
# iq

import seaborn as sns

fig, ax = plt.subplots(figsize=(20,10))

sns.set(style="ticks", palette="colorblind")

(iq_correlations

     .total_cases

     .drop('total_cases') # don't compare with myself

     .sort_values(ascending=False)

     .plot

     .barh()

)
dfiq.shape
dfiq.columns
maxlag=8

test = 'ssr_chi2test'

variables = dfsj.columns



def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    

    

    """

    data      : pandas dataframe containing the time series variables

    variables : list containing names of the time series variables.

    """

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in df.columns:

        for r in df.index:

            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)

            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]

            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)

            df.loc[r, c] = min_p_value

    df.columns = [var + '_x' for var in variables]

    df.index = [var + '_y' for var in variables]

    return df



grangers_causation_matrix(dfsj, variables)
dffsj = dfsj[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'reanalysis_dew_point_temp_k','station_min_temp_c', 'total_cases']]
dffsj.head()
from statsmodels.tsa.stattools import grangercausalitytests

maxlag=8

test = 'ssr_chi2test'

variables = dfiq.columns



def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    

    

    """

    data      : pandas dataframe containing the time series variables

    variables : list containing names of the time series variables.

    """

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in df.columns:

        for r in df.index:

            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)

            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]

            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)

            df.loc[r, c] = min_p_value

    df.columns = [var + '_x' for var in variables]

    df.index = [var + '_y' for var in variables]

    return df



grangers_causation_matrix(dfiq, variables)
dffiq = dfiq[['reanalysis_relative_humidity_percent', 'station_avg_temp_c', 'station_min_temp_c','reanalysis_min_air_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'total_cases']]
dffiq.head()
from statsmodels.tsa.vector_ar.vecm import coint_johansen



def cointegration_test(df, alpha=0.05): 

    """Perform Johanson's Cointegration Test and Report Summary"""

    out = coint_johansen(df,-1,5)

    d = {'0.90':0, '0.95':1, '0.99':2}

    traces = out.lr1

    cvts = out.cvt[:, d[str(1-alpha)]]

    def adjust(val, length= 6): return str(val).ljust(length)



    # Summary

    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)

    for col, trace, cvt in zip(df.columns, traces, cvts):

        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
cointegration_test(dffsj)
cointegration_test(dffiq)
nobs = int(len(dffsj)*.3)



dffsj_train = dffsj[:-nobs]

dffsj_test = dffsj[-nobs:]



print('30%  = ',nobs )

print('Rows , Cols for training : ', dffsj_train.shape)  

print('Rows , Cols for testing  : ', dffsj_test.shape)  
def adfuller_test(series, signif=0.05, name='', verbose=False):

    """Perform ADFuller to test for Stationarity of given series and print report"""

    r = adfuller(series, autolag='AIC')

    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}

    p_value = output['pvalue'] 

    def adjust(val, length= 6): return str(val).ljust(length)



    # Print Summary

    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)

    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')

    print(f' Significance Level    = {signif}')

    print(f' Test Statistic        = {output["test_statistic"]}')

    print(f' No. Lags Chosen       = {output["n_lags"]}')



    for key,val in r[4].items():

        print(f' Critical value {adjust(key)} = {round(val, 3)}')



    if p_value <= signif:

        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")

        print(f" => Series is Stationary.")

    else:

        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")

        print(f" => Series is Non-Stationary.") 
nobs = int(len(dffiq)*.3)



dffiq_train = dffiq[:-nobs]

dffiq_test = dffiq[-nobs:]



print('30%  = ', nobs )

print('Rows , Cols for training : ', dffiq_train.shape)  

print('Rows , Cols for testing  : ', dffiq_test.shape)
# ADF Test on each series (as column value)

for name, column in dffsj_train.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
for name, column in dffiq_train.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
model_sj = VAR(dffsj)

for i in range(1,12):

    result = model_sj.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model_sj.select_order(maxlags=3)

x.summary()
model_sj_fitted = model_sj.fit(3)

model_sj_fitted.summary()
from statsmodels.stats.stattools import durbin_watson

out = durbin_watson(model_sj_fitted.resid)



for col, val in zip(dffsj.columns, out):

    print((col), ':', round(val, 2))
# Get the lag order

lag_order = model_sj_fitted.k_ar

print(lag_order) 



# Input data for forecasting

forecastsj_input = dffsj.values[-lag_order:]

print(forecastsj_input.shape)
fc = model_sj_fitted.forecast(y = forecastsj_input, steps=280)

cols = dffsj.columns

dffsj_forecast = pd.DataFrame(fc, index=dffsj_test.index, columns=cols + '_2d')

dffsj_forecast
model_iq = VAR(dffiq)

for i in range(1,12):

    result = model_iq.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model_iq.select_order(maxlags=3)

x.summary()
model_iq_fitted = model_iq.fit(3)

model_iq_fitted.summary()
out = durbin_watson(model_iq_fitted.resid)



for col, val in zip(dffiq.columns, out):

    print((col), ':', round(val, 2))
# Get the lag order

lag_order = model_iq_fitted.k_ar

print(lag_order)



# Input data for forecasting

forecastiq_input = dffiq.values[-lag_order:]

print(forecastiq_input)
fc = model_iq_fitted.forecast(y = forecastiq_input, steps=156)

cols = dffiq.columns

dffiq_forecast = pd.DataFrame(fc, index=dffiq_test.index, columns=cols + '_2d')

dffiq_forecast
from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    maxs = np.amax(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})
print('Forecast Accuracy of: Total Cases in sj')

accuracy_prod = forecast_accuracy(dffsj_forecast['total_cases_2d'].values, dffsj_test['total_cases'])

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



print('Forecast Accuracy of: Total Cases in iq')

accuracy_prod = forecast_accuracy(dffiq_forecast['total_cases_2d'].values, dffiq_test['total_cases'])

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))
dfsj.shape
dfsj.isnull().sum()
dffsj.shape
model_sj_entire = VAR(dffsj)

for i in range(1,12):

    result = model_sj_entire.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model_sj_entire.select_order(maxlags=12)

x.summary()
model_sj_entire_fitted = model_sj_entire.fit(3)

model_sj_entire_fitted.summary()
dffsj.shape
# Get the lag order

lag_order = model_sj_entire_fitted.k_ar

print(lag_order)  #> 4



# Input data for forecasting

forecast_sj_entire_input = dffsj.values[-lag_order:]

forecast_sj_entire_input
model_iq_entire = VAR(dffiq)

for i in range(1,12):

    result = model_iq_entire.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model_iq_entire.select_order(maxlags=12)

x.summary()
model_iq_entire_fitted = model_iq_entire.fit(3)

model_iq_entire_fitted.summary()
lag_order = model_iq_entire_fitted.k_ar

print(lag_order)  



forecast_iq_entire_input = dffiq.values[-lag_order:]

forecast_iq_entire_input
fc = model_iq_entire_fitted.forecast(y=forecast_iq_entire_input, steps=156)

fc.shape
dff_test.shape
dff_iq_entire_test = dff_test[260:]
dff_iq_entire_test.set_index('week_start_date', inplace=True)

dff_iq_entire_test.sample()
cols = dffiq.columns
df_iq_entire_forecast = pd.DataFrame(fc, index=dff_iq_entire_test.index, columns=cols + '_2d')

df_iq_entire_forecast
def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    maxs = np.amax(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    return({'mae': mae, 'corr':corr})
print('Forecast Accuracy of features\n')



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['reanalysis_air_temp_k_2d'].values, dff_sj_test['reanalysis_air_temp_k'])

print('\n reanalysis_air_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['reanalysis_avg_temp_k_2d'].values, dff_sj_test['reanalysis_avg_temp_k'])

print('\n reanalysis_avg_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['reanalysis_max_air_temp_k_2d'].values, dff_sj_test['reanalysis_max_air_temp_k'])

print('\n reanalysis_max_air_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['reanalysis_min_air_temp_k_2d'].values, dff_sj_test['reanalysis_min_air_temp_k'])

print('\n reanalysis_min_air_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['reanalysis_specific_humidity_g_per_kg_2d'].values, dff_sj_test['reanalysis_specific_humidity_g_per_kg'])

print('\n reanalysis_specific_humidity_g_per_kg_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['station_avg_temp_c_2d'].values, dff_sj_test['station_avg_temp_c'])

print('\n station_avg_temp_c_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['reanalysis_dew_point_temp_k_2d'].values, dff_sj_test['reanalysis_dew_point_temp_k'])

print('\n reanalysis_dew_point_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_sj_entire_forecast['station_min_temp_c_2d'].values, dff_sj_test['station_min_temp_c'])

print('\n station_min_temp_c_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))
df_iq_entire_forecast.columns
print('Forecast Accuracy of features\n')



accuracy_prod = forecast_accuracy(df_iq_entire_forecast['reanalysis_relative_humidity_percent_2d'].values, dff_iq_entire_test['reanalysis_relative_humidity_percent'])

print('\n reanalysis_relative_humidity_percent_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_iq_entire_forecast['station_avg_temp_c_2d'].values, dff_iq_entire_test['station_avg_temp_c'])

print('\n station_avg_temp_c_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_iq_entire_forecast['station_min_temp_c_2d'].values, dff_iq_entire_test['station_min_temp_c'])

print('\n station_min_temp_c_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_iq_entire_forecast['reanalysis_min_air_temp_k_2d'].values, dff_iq_entire_test['reanalysis_min_air_temp_k'])

print('\n reanalysis_min_air_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))  



accuracy_prod = forecast_accuracy(df_iq_entire_forecast['reanalysis_dew_point_temp_k_2d'].values, dff_iq_entire_test['reanalysis_dew_point_temp_k'])

print('\n reanalysis_dew_point_temp_k_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))



accuracy_prod = forecast_accuracy(df_iq_entire_forecast['reanalysis_specific_humidity_g_per_kg_2d'].values, dff_iq_entire_test['reanalysis_specific_humidity_g_per_kg'])

print('\n reanalysis_specific_humidity_g_per_kg_2d')

for k, v in accuracy_prod.items():

    print((k), ': ', round(v,4))