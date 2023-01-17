import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

from scipy import stats

import statsmodels.api as sm

import warnings

from itertools import product

from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-poster')
# Load data

df = pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

df.head()
# Unix-time to 

df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')



# Resampling to daily frequency

df.index = df.Timestamp

df = df.resample('D').mean()



# Resampling to monthly frequency

df_month = df.resample('M').mean()



# Resampling to annual frequency

df_year = df.resample('A-DEC').mean()



# Resampling to quarterly frequency

df_Q = df.resample('Q-DEC').mean()
# PLOTS

fig = plt.figure(figsize=[15, 7])

plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)



plt.subplot(211)

plt.plot(df.Weighted_Price, '-', label='By Days')

plt.legend()



plt.subplot(212)

plt.plot(df_month.Weighted_Price, '-', label='By Months')

plt.legend()



# plt.tight_layout()

plt.show()
n_lags = 20

lags = np.arange(1, n_lags+1)

lags
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df_month.Weighted_Price, lags=lags, title = 'ACF BTC')

plt.xticks(np.arange(0, n_lags+1,2))

plt.show()
plot_pacf(df_month.Weighted_Price, lags=lags, title = 'PACF BTC')

plt.xticks(np.arange(0, n_lags+1,2))

plt.show()
df_month.head(5)
df_month.Weighted_Price[:5]
df_month['Weighted_Price_diff'] = df_month['Weighted_Price']

df_month.head(5)
df_month['Weighted_Price_diff'] = df_month.Weighted_Price_diff - df_month.Weighted_Price_diff.shift(1)

df_month.head(5)
# PLOTS

fig = plt.figure(figsize=[15, 7])

plt.suptitle('$\Delta$BTC, USD', fontsize=22)



plt.subplot(111)

plt.plot(df_month.Weighted_Price_diff, '-', label='$\Delta$BTC')

plt.legend()



# plt.tight_layout()

plt.show()
df_month.Weighted_Price[:5]
df_month.Weighted_Price_diff[:5]
df_month_without_1 = df_month.Weighted_Price_diff

df_month_without_1[:5]
df_month_without_1 = df_month_without_1.drop(df_month_without_1.index[[0]])

df_month_without_1[:5]
plot_acf(df_month_without_1, lags=lags, title = 'ACF $\Delta$BTC')

plt.xticks(np.arange(0, n_lags+1,2))

plt.show()
plot_pacf(df_month_without_1, lags=lags, title = 'PACF $\Delta$BTC')

plt.xticks(np.arange(0, n_lags+1,2))

plt.show()