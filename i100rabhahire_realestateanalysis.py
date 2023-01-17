# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_datareader import data

import datetime

from pandas import Series, DataFrame

pd.__version__



import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rc('figure', figsize=(8, 7))

mpl.__version__

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_stock = pd.read_csv('/kaggle/input/realestatedatasets/01-04-2020-TO-01-10-2020BRIGADEALLN.csv', index_col = 'Date', parse_dates = True)

df_index = pd.read_csv('/kaggle/input/realestatedatasets/NIFTY Realty_Data.csv', index_col='Date', parse_dates=True)

df_stock['diff'] = df_stock['Open Price'] - df_stock['Close Price']

df_stock.head(126)
df_index['diff'] = df_index.Open - df_index.Close

df_index.head(126)
df_stock.describe()
df_index.describe()
stock_close = df_stock['Close Price']

moving_avg = stock_close.rolling(40).mean()

moving_avg[-10:]
stock_rets = stock_close / stock_close.shift(1) - 1

stock_rets.head()
stock_close.plot(label='BRIGADE')

moving_avg.plot(label='moving_avg')

plt.legend()
index_close= df_index['Close']

moving_avg1 = index_close.rolling(40).mean()

moving_avg1[-10:]
index_rets = index_close / index_close.shift(1) - 1

index_rets.head()
index_close.plot(label='NIFTY')

moving_avg1.plot(label='moving_avg')

plt.legend()
stock_close.plot(label='BRIGADE')

moving_avg.plot(label='moving_avg_BRIGADE')

index_close.plot(label='NIFTY')

moving_avg1.plot(label='moving_avg_NIFTY')

plt.legend()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df_stock['Close Price'], df_index['Close'])

print("The Pearson Coefficient is",pearson_coef,"with a P-value of P=",p_value)