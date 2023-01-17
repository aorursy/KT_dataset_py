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
from matplotlib import pyplot
from matplotlib import style
import math 
df = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv',parse_dates=True, index_col=0)
df.head()
df[-10:]
df.describe()
# ax1 = pyplot.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
# ax2 = pyplot.subplot2grid((6,1),(5,0),rowspan=1,colspan=1)

# ax1.plot(df.index, df['Weighted_Price'])
# ax2.bar(df.index, df['Volume_(Currency)'])

# plt.show()
df['Weighted_Price'].plot()
pyplot.show()
pyplot.plot(df['Weighted_Price'])
pyplot.xlabel('time')
pyplot.ylabel('log_prices')
pyplot.title('Log_BTC_prices')

#pyplot.xscale('log')
pyplot.yscale('log')

pyplot.show()
df['Volume_(Currency)'].plot()
pyplot.show()
pyplot.plot(df['Volume_(Currency)'])
pyplot.xlabel('time')
pyplot.ylabel('log_volume')
pyplot.title('Log_USD_volume')

#pyplot.xscale('log')
pyplot.yscale('log')

pyplot.show()
df.count()
4.5*365*24*60 #Años*dias año*horas año*minuntos año