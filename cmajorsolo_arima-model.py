# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

data = data.dropna()

# data.head(5)

data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

data.head(5)
df_index = data.set_index(['Timestamp'])

df_index = df_index.sort_index(axis=1, ascending=True)

print(df_index.head())
weighted_price_data = df_index['Weighted_Price']

weighted_price_data.plot(y='Weighted_Price')
weighted_price_data.head()
data.head(5)
from pandas.plotting import autocorrelation_plot

import matplotlib.pyplot as plt



mask = (data['Timestamp'] <= '2015-01-07 22:06:00')

sub_df = data.loc[mask]

sub_df = sub_df[['Timestamp', 'Weighted_Price']]

sub_df = sub_df.set_index('Timestamp')



# dr = pd.date_range(start='2014-12-01 05:33:00', end='2014-12-02 05:29:00')

# df = pd.DataFrame(np.arange(len(dr)), index=dr, columns=["Values"])

# autocorrelation_plot(df)

# plt.show()



autocorrelation_plot(sub_df)

plt.show()
data.tail(5)
arima_data = data[['Weighted_Price']]

# arima_data.set_index('Timestamp')
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot



# model = ARIMA(arima_data, order=(2, 1, 0))

# model_fit = model.fit(disp=0)

# print(model_fit.summary())

from pandas import DataFrame



residuals = DataFrame(model_fit.resid)

residuals.plot()

pyplot.show()

residuals.plot(kind='kde')

pyplot.show()

print(residuals.describe())
data['Weighted_Price'].head(5)
from statsmodels.tsa.stattools import adfuller

from numpy import log



result = adfuller(data['Weighted_Price'][:1000])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({'figure.figsize':(9, 7), 'figure.dpi':120})



data_sub = data['Weighted_Price'][:1000]
# Original Series

fig, axes = plt.subplots(3, 2, sharex=True)

axes[0, 0].plot(data_sub)

axes[0, 0].set_title('Original Series')

plot_acf(data_sub, ax=axes[0, 1])



# 1st differencing

axes[1, 0].plot(data_sub.diff())

axes[1, 0].set_title('1st Order Differencing')

plot_acf(data_sub.diff(), ax=axes[1, 1])



# 2nd differencing

axes[2, 0].plot(data_sub.diff().diff())

axes[2, 0].set_title('2nd Order Differencing')

plot_acf(data_sub.diff().diff(), ax=axes[2, 1])



plt.show()
