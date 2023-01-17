# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas import read_csv





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/5) Recurrent Neural Network/international-airline-passengers.csv',skipfooter=5,index_col="Month")
df.tail()
df.rename(columns={'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60': 'Passengers'}, inplace=True)
df.dropna(inplace=True)
df.index
df.index = pd.to_datetime(df.index)
df.index
df['6-month-SMA'] = df['Passengers'].rolling(window=6).mean()

df['12-month-SMA'] = df['Passengers'].rolling(window=12).mean()
df.plot(figsize=(10,8));
df['EWMA-12'] = df['Passengers'].ewm(span=12).mean()

df[['Passengers','EWMA-12']].plot(figsize=(10,8));
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Passengers'],model='multiplicative')

result
result.seasonal.plot(figsize=(10,8))
result.trend.plot(figsize=(10,8))
fig =result.plot()
time_series = df['Passengers']
type(time_series)
time_series.rolling(12).mean().plot(label='12 Month Rolling Mean',figsize=(10,8))

time_series.rolling(12).std().plot(label='12 STD Mean',figsize=(10,8))

time_series.plot()

plt.legend();