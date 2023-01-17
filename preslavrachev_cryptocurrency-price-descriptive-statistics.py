# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
prices = json.load(open('../input/btc-historical-prices.json', 'r'))['Data']
df = pd.DataFrame(prices)
df.tail()
df['w_average'] = (df['open'] + df['close'] + df['high']) / 3
df['hilo_ratio'] = df['low'] / df['high']
df['daily_pct_change'] = df['w_average'].pct_change()
df['365d_rolling_daily_pct_change'] = df['daily_pct_change'].rolling(365).mean()

# Add a 14-day stocahstic
wa_rolling_14d = df['w_average'].rolling(14)
rolling_14d_min, rolling_14d_max = wa_rolling_14d.min(), wa_rolling_14d.max()
df['14d_stochastic_k'] = (df['w_average'] - rolling_14d_min) / (rolling_14d_max - rolling_14d_min)

df['vf_daily_pct_change'] = df['volumefrom'].pct_change()
df['vt_daily_pct_change'] = df['volumeto'].pct_change()
df['time'] = pd.to_datetime(df['time'], unit='s')
df.tail(20)

#df['daily_pct_change'].describe()
#sns.distplot(df['daily_pct_change'].dropna())

#df.plot.line(x='time', y='daily_pct_change')
#df.plot.line(x='time', y='365d_rolling_daily_pct_change')
#df.tail(365).plot.line(x='time', y='365d_rolling_daily_pct_change')

#df.tail(30).plot.line(x='time', y=['vf_daily_pct_change', 'vt_daily_pct_change'])

#df.tail(365).plot.scatter(x='14d_stochastic_k', y='vf_daily_pct_change')
df['label'] = df['daily_pct_change'] > df['daily_pct_change'].rolling(30).mean() + df['daily_pct_change'].rolling(30).std()
df[df['label'] == True].corr()