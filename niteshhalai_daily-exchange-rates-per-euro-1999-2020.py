# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/euro-exchange-daily-rates-19992020/euro-daily-hist_1999_2020.csv')
data
data.set_index('Period\\Unit:', inplace=True)

data.index = pd.to_datetime(data.index)

data = data.resample('1M').mean()
data
from statsmodels.tsa.filters.hp_filter import hpfilter

price_cycle, price_trend = hpfilter(data['[Turkish lira ]'], lamb = 129600)

data['trend'] = price_trend

data[['trend', '[Turkish lira ]']].plot(figsize=(12,5))
price_cycle, price_trend = hpfilter(data['[Iceland krona ]'], lamb = 129600)

data['trend'] = price_trend

data[['trend', '[Iceland krona ]']].plot(figsize=(12,5))
price_cycle, price_trend = hpfilter(data['[Romanian leu ]'], lamb = 129600)

data['trend'] = price_trend

data[['trend', '[Romanian leu ]']].plot(figsize=(12,5))