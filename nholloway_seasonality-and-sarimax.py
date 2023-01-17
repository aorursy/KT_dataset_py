import math

import itertools

import pandas as pd

import numpy as np

from scipy.stats import boxcox, kurtosis, skew

from sklearn.metrics import mean_squared_error, mean_absolute_error 

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARIMAResults

from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.statespace.sarimax import SARIMAX



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight') 

plt.rcParams['xtick.labelsize'] = 20

plt.rcParams['ytick.labelsize'] = 20



%matplotlib inline
df = pd.read_csv('../input/candy_production.csv')

df = df.rename(columns={'observation_date': 'date', 'IPG3113N':'production'})

df.index = pd.DatetimeIndex(data= df.date)

df = df.drop(columns=['date'])

df.head()
df.plot(figsize=(20, 10), fontsize=20)

plt.title('Candy Production from 1972-2017', fontsize=30)

plt.show()
df['bc_production'], lamb = boxcox(df.production)

df['df1_production'] = df['bc_production'].diff()

df['df_production'] = df['production'].diff()

fig = plt.figure(figsize=(20,40))



bc = plt.subplot(411)

bc.plot(df.bc_production)

bc.title.set_text('Box-Cox Transform')

df1 = plt.subplot(412)

df1.plot(df.df1_production)

df1.title.set_text('First-Order Transform w/ Box-Cox')

df2 = plt.subplot(413)

df2.plot(df.df_production)

df2.title.set_text('First-Order Transform w/o Box-Cox')



df.bc_production.dropna(inplace=True)

df.df1_production.dropna(inplace=True)

df.df_production.dropna(inplace=True)



print(f'Lambda Value {lamb}')
f_acf = plot_acf(df['df_production'], lags=50)

f_pacf = plot_pacf(df['df_production'], lags=50, method='ols')

f_acf.set_figheight(10)

f_acf.set_figwidth(15)

f_pacf.set_figheight(10)

f_pacf.set_figwidth(15)

plt.show()
split_date = '2008-12-01'

train = df['production'].loc[:split_date]

test = df['production'].loc[split_date:]

train.plot(figsize=(20, 10), fontsize=20)

plt.title('Candy Production Train/Test Split', fontsize=30)

test.plot()

plt.show()
model = SARIMAX(train, freq='MS', order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))

model_fit = model.fit(disp=False) 
fcast_len = len(test)

fcast = model_fit.forecast(fcast_len)

mse = mean_squared_error(test, fcast)

rmse = np.sqrt(mse)

mae = mean_absolute_error(test, fcast)

plt.figure(figsize=(20, 10))

plt.title('Candy Production Forecast', fontsize=30)

plt.plot(train, label='Train')

plt.plot(fcast, label='Forecast')

plt.plot(test, label='Test')



print(f'Mean Squared Error: {mse}')

print(f'Root Mean Squared Error: {rmse}')

print(f'Mean Absolute Error: {mae}')

plt.legend(fontsize=25)

plt.show()
def rolling_forecast(train, test, order, season):

    history = [x for x in train]

    model = SARIMAX(history, order= order, seasonal_order= season)

    model_fit = model.fit(disp=False)

    predictions = []

    results = {}

    yhat = model_fit.forecast()[0]



    predictions.append(yhat)

    history.append(test[0])

    for i in range(1, len(test)):

        model = SARIMAX(history, order= order, seasonal_order= season)

        model_fit = model.fit(disp=False)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        obs = test[i]

        history.append(obs)

    mse = mean_squared_error(test, predictions)

    mae = mean_absolute_error(test, predictions)

    rmse = math.sqrt(mse)

    predictions = pd.Series(predictions, index=test.index)

    results['predictions'] = predictions

    results['mse'] = mse

    results['rmse'] = rmse

    results['mae'] = mae

    return results
rolling_fcast = rolling_forecast(train, test, (1, 1, 1), (1, 0, 0, 12))
plt.figure(figsize=(20, 10))

plt.title('Candy Production Rolling Forecast', fontsize=30)

plt.plot(train, label='Train')

plt.plot(rolling_fcast['predictions'], label='Forecast')

plt.plot(test, label='Test')



print(f'Mean Squared Error: {rolling_fcast["mse"]}')

print(f'Root Mean Squared Error: {rolling_fcast["rmse"]}')

print(f'Mean Absolute Error: {rolling_fcast["mae"]}')

plt.legend(fontsize=25)

plt.show()