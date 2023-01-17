# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import dataset

ds_cryptoprices = pd.read_csv('../input/crypto_prices.csv')



ds_cryptoprices.info()
ds_cryptoprices.corr()


#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(ds_cryptoprices.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
ds_cryptoprices.head(10)
ds_cryptoprices.columns

# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

ds_cryptoprices.Close.plot(kind = 'line', color = 'g',label = 'Close',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

ds_cryptoprices.Volume.plot(color = 'r',label = 'Volume',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Close axis')              # label = name of label

plt.ylabel('Volume axis')

plt.title('Crypto Prices Line Plot')            # title = title of plot

plt.show()

#Scatter

ds_cryptoprices.columns

plt.scatter(ds_cryptoprices.Close, ds_cryptoprices.Volume, color ="red", alpha=0.5)

plt.xlabel("Close")

plt.ylabel("Volume")

plt.show()
# Scatter Plot 

# x = attack, y = defense

ds_cryptoprices.plot(kind='scatter', x='Close', y='Volume', alpha = 0.5, color = 'red')

plt.xlabel('Close')              # label = name of label

plt.ylabel('Volume')

plt.title('Crypto Prices Line Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

ds_cryptoprices.Close.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.clf()

plt.show()
# clf() = cleans it up again you can start a fresh

ds_cryptoprices.Close.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#series

sr_cryptoprices = ds_cryptoprices['Close']

print(type(sr_cryptoprices))

#dataframe

df_cryptoprices = ds_cryptoprices[['Close']]

print(type(df_cryptoprices))
x = ds_cryptoprices['Volume']>1

ds_cryptoprices[x]
# 2 - Filtering pandas with logical_and

ds_cryptoprices[np.logical_and(ds_cryptoprices['Low']>1, ds_cryptoprices['Volume']>1)]
# Same as above filtering method

ds_cryptoprices[(ds_cryptoprices['Low']>1) & (ds_cryptoprices['Volume']>1)]
# For pandas we can achieve index and value

for index,value in ds_cryptoprices[['Low']][0:10].iterrows():

    print(index," : ",value)