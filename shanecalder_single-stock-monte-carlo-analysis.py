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
from pandas_datareader import data as wb

import matplotlib.pyplot as plt

from scipy.stats import norm

%matplotlib inline
AMZN = pd.read_csv('/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AMZN_data.csv')
#AMZN Amazon 

AMZN_close = AMZN[['date','close']].copy()
AMZN_close.head()
AMZN_close.tail()
#log_returns = np.log(1 + AMZN.pct_change())



AMZN_Log_returns = np.log(AMZN_close.close).pct_change()
AMZN_Log_returns.rename(columns = {list(AMZN_Log_returns)[1]:'pct_change'}, inplace=True)
df_AMZN = pd.concat([AMZN_close, AMZN_Log_returns], axis = 1)

df_AMZN.head()
#Need to rename last column (work around as data is not live)

df_AMZN.rename(columns = {list(df_AMZN)[2]:'pct_change'}, inplace=True)

df_AMZN.head()
#Lets look at closing price (if you use Google or Yahoo use the adjusted close)

AMZN_close.plot(figsize=(14,8))
#Lets look at the percent change 

AMZN_Log_returns.plot(figsize=(14,8))
u = AMZN_Log_returns.mean()

u
var = AMZN_Log_returns.var()

var
stdev = AMZN_Log_returns.std()

stdev
drift = u - (0.5 * var)

drift
np.array(drift)
#set up a 95% chance of occurance

norm.ppf(0.95)
x = np.random.rand(10,2)

x
norm.ppf(x)
Z = norm.ppf(np.random.rand(10,2))

Z
# Forcast projections

"""

t_intervals is the amount of days you want to forcast

interactions is the amount of forcasted projections 

"""

t_intervals = 1000

interations = 10
daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, interations)))

daily_returns
AMZN_1 = AMZN[['close']].copy()

s_zero = AMZN_1.iloc[-1]

s_zero
price_list = np.zeros_like(daily_returns)
price_list
price_list[0] = s_zero

price_list
for t in range(1, t_intervals):

    price_list[t] = price_list[t - 1] * daily_returns[t]
price_list
# to get a good idea of projections run multiple times

plt.figure(figsize=(14,8))

plt.title('Amazon Stock Price Predictions')

plt.xlabel('Future Days Traded')

plt.ylabel('Stock Price US')

plt.plot(price_list)