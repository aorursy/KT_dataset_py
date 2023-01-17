# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

from pandas import DataFrame

from pandas import concat

from pandas.plotting import register_matplotlib_converters

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.tsa.api as smt

from statsmodels.tsa.arima_model import ARIMA as ARIMA

sns.set()

plt.rcParams["figure.figsize"] = [16,9]



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')

df['Month']=pd.to_datetime(df['Month'], infer_datetime_format=True)

indf=df.set_index(['Month'])

indf.head()

df.head()
series = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv', header=0, index_col=0,

parse_dates=True, squeeze=True)

temps = DataFrame(series.values)

dataframe = concat([temps.shift(1), temps], axis=1)

dataframe.columns = ['t','t+1']

print(dataframe.head(5))
df.shape
df.info()
df.describe()
plt.xlabel('Date')

plt.ylabel('Number of passengers')

plt.plot(series)
series.hist()

plt.show()
from pandas import DataFrame

from pandas import Grouper

groups = series.groupby(Grouper(freq='A'))

years = DataFrame()

for name, group in groups:

    years[name.year] = group.values

years.boxplot()

plt.show()
temps = DataFrame(series.values)

shifted = temps.shift(1)

window = shifted.rolling(window=2)

means = window.mean()

dataframe = concat([means, temps], axis=1)

dataframe.columns = ['mean(t-1,t)', 't+1']

print(dataframe.head(5))
movingaverage=series.rolling(window=12).mean()

movingstd=series.rolling(window=12).std()

plt.plot(series)

plt.plot(movingaverage, color='red')

plt.plot(movingstd, color='black')
from statsmodels.tsa.stattools import adfuller

def stationarity_check(ts):

    

    roll_mean = ts.rolling(window=12).mean()

    movingstd = ts.rolling(window=12).std()



    #Plot rolling statistics:

    orig = plt.plot(ts, color='blue',label='Original')

    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')

    std = plt.plot(movingstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)



    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(ts, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
stationarity_check(series)
result = seasonal_decompose(series, model='multiplicative')

result.plot()

plt.show()
stationarity_check(series)
log_series = np.log(series)

log_series.head()
stationarity_check(log_series)
series_diff = series.diff()

series_diff = series_diff.dropna()

stationarity_check(series_diff)
log_series_diff = log_series.diff()

log_series_diff = log_series_diff.dropna()

stationarity_check(log_series_diff)
movingAverage = log_series.rolling(window=12).mean()

ma_log_series=log_series-movingAverage

ma_log_series = ma_log_series.dropna()

stationarity_check(ma_log_series)
shift_log_series = log_series - log_series.shift()

shift_log_series = shift_log_series.dropna()

stationarity_check(shift_log_series)
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)

plt.show()
def acfpacf(y, lags=None, figsize=(12, 7), style='bmh'):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        p_value = adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
acfpacf(log_series_diff)
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(log_series_diff, nlags=20)

lag_pacf = pacf(log_series_diff, nlags=20, method='ols')



plt.subplot(121)

plt.plot(lag_acf)

plt.axhline(y=0, linestyle='--', color='gray')

plt.axhline(y=-1.96/np.sqrt(len(log_series_diff)), linestyle='--', color='gray')

plt.axhline(y=1.96/np.sqrt(len(log_series_diff)), linestyle='--', color='gray')

plt.title('Autocorrelation Function')            



#Plot PACF

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0, linestyle='--', color='gray')

plt.axhline(y=-1.96/np.sqrt(len(log_series_diff)), linestyle='--', color='gray')

plt.axhline(y=1.96/np.sqrt(len(log_series_diff)), linestyle='--', color='gray')

plt.title('Partial Autocorrelation Function')

            

plt.tight_layout()
model = ARIMA(log_series, order=(1,1,1))

results_ARIMA = model.fit(disp=-1)

plt.plot(log_series_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

print('Plotting ARIMA model')

print(results_ARIMA.summary())
residuals = DataFrame(results_ARIMA.resid)

residuals.plot(kind='kde')

plt.show()
print(residuals.describe())
model = ARIMA(log_series, order=(2,1,2))

results_ARIMA = model.fit(disp=-1)

plt.plot(log_series_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

print('Plotting ARIMA model')

print(results_ARIMA.summary())
residuals = DataFrame(results_ARIMA.resid)

residuals.plot(kind='kde')

plt.show()
print(residuals.describe())
pred = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(pred.head())
pred_sum = pred.cumsum()

print(pred_sum)
pred_log = pd.Series(log_series.iloc[0], index=log_series.index)

pred_log = pred_log.add(pred_sum, fill_value=0)

pred_log.head()
final_pred = np.exp(pred_log)

plt.plot(series)

plt.plot(final_pred)