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
df = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")

import matplotlib.pyplot as plt

import seaborn as sns
df.head()
df.tail()



#wanted to see last values in df
df.columns
df.info()

#the fact that the values are float, it can be plotted in line graph
#correlation map

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, linewidths=.8, fmt='.1f', ax=ax)

plt.show()



#open, high, low,close and weighted_price correlate with each other

#Volume_Btc and Volume_Currency correlate with the rest the least
plt.figure(figsize=(20,4))

df.Open.plot(kind='line', color='r', label='Open', alpha=0.5, linewidth=5, grid=True, linestyle=':')

df.High.plot(color='g', label='High', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right') #legend put label into plot

plt.xlabel('Time')

plt.ylabel('price at the start of the time window')

plt.title('Line plot')

plt.show()



#open and high values have the same trend
df.plot(kind='scatter', x='Volume_(BTC)', y='Volume_(Currency)', alpha=1.0, color='red')

plt.xlabel('volume_BTC')

plt.ylabel('volume_Currency')

plt.title('BTC vs Currency Scatter Plot')



#it looks like there is no correlation between the too.

#how the points are scatter suggests that while one is
print(type(df))

df_open_BTC = df[['Open', 'Volume_(BTC)']]

print(df_open_BTC)



#if ope price will correlate with the volume of BTC? Therefore, i separated those columns from the rest. 
df_open_BTC.plot(kind = 'scatter', x = 'Open', y = 'Volume_(BTC)', alpha=0.8, color = 'blue')



#i wanted to see if there is any correlation between two columns 

#beginning of the plot looks different than the rest. that is why, i wanted to zoom in that region

#open > 1000 and volume > 2000
x = df[np.logical_and(df['Open']<1000, df['Volume_(BTC)']<2000)]

#plt.figure(figsize=(15,15)), why didnt it work here? in the previous code, i used it to resize the plot

x.plot(kind = 'scatter', x = 'Open', y = 'Volume_(BTC)', alpha=0.8, color = 'green')



# i zoomed in the details of these two columns
# checking index and value of first 20 values from Open columns

for index,value in x[['Open']][0:20].iterrows():

    print(index, ":", value)