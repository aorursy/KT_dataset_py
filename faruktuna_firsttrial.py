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
data =pd.read_csv('../input/2015.csv')
data.info()

data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.tail(15)
data.columns
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='Happiness Score', y='Happiness Rank',alpha = 0.5,color = 'red')

plt.xlabel('Happiness Score')              # label = name of label

plt.ylabel('Happiness Rank')

plt.title('Rank - Score Plot')            # title = title of plot
data.plot(kind= 'scatter', x = 'Happiness Score' , y =  'Health (Life Expectancy)' , alpha = 0.5 , color= 'b')

plt.xlabel('HS')              

plt.ylabel('HLE')

plt.title('Happiness Score and HLE relation') 
# Histogram

# bins = number of bar in figure

data.Generosity.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.Family.plot(kind = 'hist',bins =25 ,figsize = (12,12))

plt.show()
series = data['Family']        # data['Defense'] = series

print(type(series))

data_frame = data[['Family']]  # data[['Defense']] = data frame

print(type(data_frame))
# 2 - Filtering pandas with logical_and



data[np.logical_and(data['Family']<1, data['Happiness Score']>7 )]
data[(data['Family']<1) & (data['Happiness Score']>7 )]
c=data['Country']

for i in c:

    print('i is ', i)

print('')



for index, value in enumerate(c):

    print(index,":",value)

print('')



for index, value in data[['Family','Generosity','Happiness Score']][0:10].iterrows():

    print(index,":",value)

x=data['Generosity']>0.4

data[x]

