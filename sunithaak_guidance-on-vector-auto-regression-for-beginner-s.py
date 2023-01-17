import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Import Statsmodels

from statsmodels.tsa.api import VAR

from statsmodels.tsa.stattools import adfuller

from statsmodels.tools.eval_measures import rmse, aic
filepath = '../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv'

df = pd.read_csv(filepath)



print(df.shape)  # (123, 8)

df.tail()
df = df.dropna(axis=1)
df.tail()
df.shape

cor = df.corr()

#Correlation with output variable

cor_target = abs(cor["Price"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.98]

relevant_features
df = df[[ 'BritishVirginIslands_total_cases' , 'Cyprus_total_deaths',

        'Grenada_total_cases', 'Guyana_total_deaths'   , 'Niger_total_cases' ,

         'Singapore_total_deaths', 'SriLanka_total_deaths','Vatican_total_cases','Price'] ]
df.shape


from statsmodels.tsa.stattools import grangercausalitytests

maxlag=12

test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    



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



grangers_causation_matrix(df, variables = df.columns)           
nobs = 30

df_train, df_test = df[0:-nobs], df[-nobs:]



# Check size

print(df_train.shape)  # (119, 8)

print(df_test.shape)  # (4, 8)
test =df_test[['Price']]
def adfuller_test(series, signif=0.05, name='', verbose=False):

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
# 1st difference

df_differenced = df_train.diff().dropna()
# ADF Test on each column of 1st Differences Dataframe

for name, column in df_differenced.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
# Second Differencing

df_differenced = df_differenced.diff().dropna()
# ADF Test on each column of 2nd Differences Dataframe

for name, column in df_differenced.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
# third Differencing

df_differenced = df_differenced.diff().dropna()
# ADF Test on each column of 3rd Differences Dataframe

for name, column in df_differenced.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
model = VAR(df_differenced)

for i in [1,2,3,4,5]:

    result = model.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model.select_order(maxlags=5)

x.summary()
model_fitted = model.fit()

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

df_forecast
def invert_transformation(df_train, df_forecast, second_diff=False):

    df_fc = df_forecast.copy()

    columns = df_train.columns

    for col in columns:        

        # Roll back 2nd Diff

        if second_diff:

            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()

        # Roll back 1st Diff

        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()

    return df_fc
df_results = invert_transformation(df_train, df_forecast, second_diff=True)        

predict = df_results['Price_forecast']
len(predict)
import math

from sklearn.metrics import mean_squared_error

math.sqrt( mean_squared_error(test,predict))
import matplotlib.pyplot as plt

# zoom plot

plt.figure(figsize=(20,10))

plt.plot(test)

plt.plot(predict, color='green')

plt.title('Actual Vs Predicted')

plt.show()
