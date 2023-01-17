# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import warnings

train1 = pd.read_csv("/kaggle/input/all-buildings/train.csv",index_col="timestamp", parse_dates = True, dayfirst=True)

train1.info()



df = train1[train1.building_number==5]

df = df.drop(['building_number'],axis=1)

df.head()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Import Statsmodels

from statsmodels.tsa.api import VAR

from statsmodels.tsa.stattools import adfuller

from statsmodels.tools.eval_measures import rmse, aic
# Plot

fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(10,10))

for i, ax in enumerate(axes.flatten()):

    data = df[df.columns[i]]

    ax.plot(data, color='red', linewidth=1)

    # Decorations

    ax.set_title(df.columns[i])

    ax.xaxis.set_ticks_position('none')

    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)

    ax.tick_params(labelsize=6)



plt.tight_layout();
nobs = 5760

df_train, df_test = df[0:-nobs], df[-nobs:]



# Check size

print(df_train.shape)  # (119, 8)

print(df_test.shape)  # (4, 8)
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
# ADF Test on each column

for name, column in df_train.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
#df_differenced = df_train.diff().dropna()

#df_differenced = df_train.diff().dropna()

df_differenced = df_train.copy()

# ADF Test on each column of 2nd Differences Dataframe

for name, column in df_differenced.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
model = VAR(df_differenced)



for i in [1,2,3,4,5,6,7,8,9]:

    result = model.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model.select_order(maxlags=20)

x.summary()
model_fitted = model.fit(672)

model_fitted.summary()
# Get the lag order

lag_order = model_fitted.k_ar

print(lag_order)  #> 4



# Input data for forecasting

forecast_input = df_differenced.values[-lag_order:]

forecast_input
# Forecast

fc = model_fitted.forecast(y=forecast_input, steps=nobs)

df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')

df_forecast = df_forecast.rename(columns={"main_meter_2d": "main_meter_forecast", "sub_meter_1_2d": "sub_meter_1_forecast","sub_meter_2_2d":"sub_meter_2_forecast"})
df_results = df_forecast

df_results
fig, axes = plt.subplots(nrows=3, ncols=1, dpi=150, figsize=(10,10))

for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):

    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)

    df_test[col][-nobs:].plot(legend=True, ax=ax);

    ax.set_title(col + ": Forecast vs Actuals")

    ax.xaxis.set_ticks_position('none')

    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)

    ax.tick_params(labelsize=6)



plt.tight_layout();
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



print('Forecast Accuracy of: main_meter')

accuracy_prod = forecast_accuracy(df_results['main_meter_forecast'].values, df_test['main_meter'])

for k, v in accuracy_prod.items():

    print(k, ': ', round(v,4))



print('\nForecast Accuracy of: sub_meter_1')

accuracy_prod = forecast_accuracy(df_results['sub_meter_1_forecast'].values, df_test['sub_meter_1'])

for k, v in accuracy_prod.items():

    print(k, ': ', round(v,4))



print('\nForecast Accuracy of: ulc')

accuracy_prod = forecast_accuracy(df_results['sub_meter_1_forecast'].values, df_test['sub_meter_1'])

for k, v in accuracy_prod.items():

    print(k, ': ', round(v,4))


