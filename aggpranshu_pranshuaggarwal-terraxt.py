

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won'tbe saved outside of the current session



import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib

import pmdarima as pm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import traceback

import datetime
matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
series = pd.read_csv('../input/into-the-future/train.csv')

series['time'] = pd.to_datetime(series['time'], format="%Y/%m/%d %H:%M:%S")

series.plot(figsize=(15, 6), x = 'time', y = 'feature_1')



plt.show()

#There is a presence of trend in the time-series (Downward trend)



#ACF and PACF plots to check seasonality and estimate the order of AR,MA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(series['feature_1'], lags=20)

matplotlib.pyplot.show()

plot_pacf(series['feature_1'], lags=20)

matplotlib.pyplot.show()
series['time'].min(), series['time'].max()
ts_series = series.set_index('time')
#Making the series stationary by taking the first order difference (ARIMAX requires series to be stationary (free of trend and seasonality))

ts_diff = ts_series - ts_series.shift(1)

ts_diff = ts_diff.dropna()
ts_diff.plot(figsize=(15, 6), y = 'feature_1')

plt.show()
# Creating a grid of p,d,q to test different ARIMAX models. We will select the one with least AIC

p = range(0, 4)

d = range(0,2)

q = range(0, 4)

pdq = list(itertools.product(p, d, q))
for param in pdq:

    try:

        mod = sm.tsa.statespace.SARIMAX(endog = ts_series['feature_2'],

                                            order=param,

                                            exog = ts_series['feature_1']

                                            )

        results = mod.fit()

        print('ARIMA{} - AIC:{}'.format(param, results.aic))

    except:

        traceback.print_exc()
# p=1, d=1, q=3 has the least AIC values amongst all

mod = sm.tsa.statespace.SARIMAX(endog = ts_series['feature_2'],

                                            order=(1,1,3),

                                            exog = ts_series['feature_1']

                                            )
result = mod.fit()
series_test = pd.read_csv('../input/into-the-future/test.csv')

exog_fc = series_test.loc[:,('feature_1')]

series_test['time'] = pd.to_datetime(series_test['time'], format="%Y/%m/%d %H:%M:%S")



forecast = result.forecast(375, exog=exog_fc)