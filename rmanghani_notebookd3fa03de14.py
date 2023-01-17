# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.stats.api import ols



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_f=pd.read_csv("../input/fundamentals.csv")

#res = ols(y=data_f['Ticker Symbol'], x=data_f)

#res

#len(data_f.columns)

x=data_f.groupby(['Ticker Symbol'])['Gross Profit','Net Income','Accounts Payable','Accounts Receivable','Investments','Net Cash Flow','Total Assets','Total Equity','Total Revenue']

x.sum()



#data_f['Ticker Symbol','Gross Profit']

#data_f.columns

#data_f.dtypes

#symbol=data_f['Ticker Symbol'].unique()

#s=list(symbol)

#for i in range(len(s)):

#    print(s[i])

#sns.pairplot(data_f)

#plt.show()



# Any results you write to the current directory are saved as output.
data_price=pd.read_csv("../input/prices.csv")

#data_price.groupby(['symbol'])['open','high','low','close']



del data_price['date']

#data_price.columns

df=data_price[data_price['symbol']=='AAP']

df

sns.pairplot(df)

plt.show()



#data_price_s=pd.read_csv("../input/prices-split-adjusted.csv")

#data_price_s
#df

res = ols(y=df['close'], x=df['open'])

res