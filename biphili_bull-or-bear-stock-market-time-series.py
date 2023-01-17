# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np 

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from datetime import datetime

#from __future__ import division
GOOG = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/goog.us.txt')

#data['Date'] = pd.to_datetime(data['Date'])

#data = data.set_index('Date')

#print(data.index.min(), data.index.max())

GOOG.head()
GOOG['Date']=pd.to_datetime(GOOG['Date'])
GOOG.describe().T
GOOG.plot(x='Date', y='Close',legend=True,figsize=(10,4))

plt.ioff()
title='VOLUME TRADED'

ylabel='Volume'

xlabel='Time'

ax=GOOG.plot(x='Date', y='Volume',legend=True,figsize=(10,4));

ax.autoscale(axis='x',tight=True)  # use both if want to scale both axis

ax.set(xlabel=xlabel,ylabel=ylabel)

plt.ioff()
GOOG['Close_10']=GOOG['Close'].rolling(10).mean()

GOOG['Close_50']=GOOG['Close'].rolling(50).mean()

ax=GOOG.plot(x='Date',y='Close',title='GOOG Close Price',figsize=(10,4))

GOOG.plot(x='Date',y='Close_10',color='red',ax=ax)

GOOG.plot(x='Date',y='Close_50',color='k',ax=ax)

plt.ioff()
GOOG['Daily Return']=GOOG['Close'].pct_change()

GOOG['Daily Return'].plot(figsize=(15,4),legend=True,linestyle='--',marker='o')

plt.ioff()
sns.distplot(GOOG['Daily Return'].dropna(),bins=2000,color='purple')

plt.ioff()