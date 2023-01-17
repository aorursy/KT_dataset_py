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
df = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')

df.head()
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool
f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df.corr(), vmin=0, vmax=1, cmap="YlGnBu", center=None, robust=False, annot=True, fmt='.1g', annot_kws=None, linewidths=0.2, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None)

plt.show()
df.head(10)
df.columns
df.rename(columns={'Volume_(BTC)':'Volume_BTC'}, inplace=True)
df.columns
df.sort_values(by=['Weighted_Price'], ascending=False)

df.Open.plot(kind = 'line', color = 'g',label = 'Open',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.Close.plot(color = 'r',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = ':')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('numbers')              # label = name of label

plt.ylabel('Open-Close')

plt.title('Line Plot')            # title = title of plot

plt.show()
df.plot(kind='scatter', x='Volume_BTC', y='Weighted_Price',alpha = 0.5,color = 'red')

plt.xlabel('Volume_BTC')              # label = name of label

plt.ylabel('Weighted_Price')

plt.title('Volume BTC vs Weighted Price')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

df.Weighted_Price.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()