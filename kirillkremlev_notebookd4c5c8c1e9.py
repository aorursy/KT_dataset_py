# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

!pip install yfinance

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import yfinance as yf

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import yfinance as yf

import pandas as pd

import numpy as np



%matplotlib inline



import matplotlib.pyplot as plt
t = ['F', 'GM','DAI.DE','VWAGY','TM']

a = {}

i = 0

for str in t:

    i = i + 1

    stock = yf.Ticker(str)

    div = stock.history(start="2018-01-01")

    #print( div[div['Stock Splits'] != 0])

    stock_div = div[div['Dividends'] != 0]

    stock_div = stock_div.copy()

    stock_div['div_per'] = stock_div['Dividends']/stock_div['Close']*100

    #print(stock_div[['Dividends']])

    stock = stock_div[['div_per']]

    stock = stock.copy()

    stock.rename(columns={ stock.columns[0]: str }, inplace = True)

    a[str] = stock

    #print(frame)
frames = []

for str in t:

    frames.append(a[str])

#print(frames)

result = pd.concat(frames, axis=1, sort=False)

result = result.fillna(0.02)
ax = result.plot(use_index=True, kind='bar', figsize=(20,10))

ax.set_xticklabels(result.index.strftime('%d.%m.%Y'))

ax.set_ylabel('Percetage')

ax.set_xlabel('')

ax.minorticks_on()

#ax.grid('on', which='minor', axis='x' )

ax.grid('on', which='major', axis='x' )

ax.grid('on', which='minor', axis='y' )

ax.grid('on', which='major', axis='y' )

plt.show()
data = yf.download(t, start="2020-01-01", end="2020-10-14",interval = "1d",)
data_close = data['Close']

data_close = data_close.copy()

for str in t:

    data_close[str] = (data_close[str]-data_close[str].loc['2020-01-02']) / data_close[str].loc['2020-01-02']
ax = data_close.plot(use_index=True, figsize=(40,20), linestyle='-', marker='o')



#ax.set_xticklabels(data.index.strftime('%d.%m.%Y'))

ax.set_ylabel('Percetage of price since 02.01.2020',fontsize=25)

ax.set_xlabel('')

#ax.minorticks_on()

#ax.grid('on', which='minor', axis='x' )

ax.grid('on', which='major', axis='x' )

#ax.grid('on', which='minor', axis='y' )

#ax.grid('on', which='major', axis='y' )

#plt.xticks(data.index,rotation=90)

plt.text(0.5, 0.95,'Recovery of auto stocks during Covid-2019', horizontalalignment='center',

         verticalalignment='center', transform=ax.transAxes, fontsize=40)

plt.show()