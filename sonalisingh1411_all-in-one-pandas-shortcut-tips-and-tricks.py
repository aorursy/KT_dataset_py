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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
stocks = pd.read_csv('../input/stockscsv/stocks.csv')

stocks
ser = stocks.groupby(['Symbol', 'Date']).Close.mean()

ser
ser.index
ser.unstack()
df = stocks.pivot_table(values='Close', index='Symbol', columns='Date')

df
ser
ser.loc['AAPL']
ser.loc['AAPL', '2016-10-03']
ser.loc[:, '2016-10-03']
df
df.loc['AAPL']
df.loc['AAPL', '2016-10-03']
df.loc[:, '2016-10-03']
stocks.set_index(['Symbol', 'Date'], inplace=True)

stocks
stocks.index
stocks.sort_index(inplace=True)

stocks
stocks.loc['AAPL']
stocks.loc[('AAPL', '2016-10-03'), :]
stocks.loc[('AAPL', '2016-10-03'), 'Close']
stocks.loc[['AAPL', 'MSFT'], :]
stocks.loc[(['AAPL', 'MSFT'], '2016-10-03'), :]
stocks.loc[(['AAPL', 'MSFT'], '2016-10-03'), 'Close']
stocks.loc[('AAPL', ['2016-10-03', '2016-10-04']), 'Close']
stocks.loc[(slice(None), ['2016-10-03', '2016-10-04']), :]
close = pd.read_csv('../input/stockscsv/stocks.csv', usecols=[0, 1, 3], index_col=['Symbol', 'Date']).sort_index()

close
volume = pd.read_csv('../input/stockscsv/stocks.csv', usecols=[0, 2, 3], index_col=['Symbol', 'Date']).sort_index()

volume
both = pd.merge(close, volume, left_index=True, right_index=True)

both
both.reset_index()
drinks = pd.read_csv('../input/drinks/drinks.csv')
drinks.info(memory_usage='deep')
pd.read_csv('../input/stock-abc/stocks1.csv')
pd.read_csv('../input/stock-abc/stocks2.csv')
pd.read_csv('../input/stock-abc/stocks3.csv')
from glob import glob

stock_files = sorted(glob('../input/stock-abc/stocks*.csv'))

stock_files
pd.concat((pd.read_csv(file) for file in stock_files))
#Alternate method

pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)
pd.read_csv('../input/all-in-one/drinks1.csv').head()
pd.read_csv('../input/all-in-one/drinks2.csv').head()
drink_files = sorted(glob('../input/all-in-one/drinks*.csv'))
pd.concat((pd.read_csv(file) for file in drink_files), axis='columns').head()
df = pd.DataFrame({'name':['John Arthur Doe', 'Jane Ann Smith'],

                   'location':['Los Angeles, CA', 'Washington, DC']})

df
df.name.str.split(' ', expand=True)
df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)

df
df.location.str.split(', ', expand=True)
df['city'] = df.location.str.split(', ', expand=True)[0]

df
df = pd.DataFrame({'col_one':['a', 'b', 'c'], 'col_two':[[10, 40], [20, 50], [30, 60]]})

df
df_new = df.col_two.apply(pd.Series)

df_new
pd.concat([df, df_new], axis='columns')
stocks
format_dict = {'Date':'{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}
stocks.style.format(format_dict)
(stocks.style.format(format_dict)

 .hide_index()

 .highlight_min('Close', color='red')

 .highlight_max('Close', color='lightgreen')

)
(stocks.style.format(format_dict)

 .hide_index()

 .background_gradient(subset='Volume', cmap='Blues')

)
(stocks.style.format(format_dict)

 .hide_index()

 .bar('Volume', color='lightblue', align='zero')

 .set_caption('Stock Prices from October 2016')

)
import pandas_profiling
pandas_profiling.ProfileReport(drinks)