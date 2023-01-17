# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(16,16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Deaths.plot(kind = 'line', color = 'b',label = 'Deaths',linewidth=3,alpha = 0.9,grid = True,linestyle = ':')

data.Confirmed.plot(color = 'r',label = 'Confirmed',linewidth=2, alpha = 0.6,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Confirmed, y = Deaths

data.plot(kind='scatter', x='Confirmed', y='Deaths',alpha = 0.5,color = 'red')

plt.xlabel('Confirmed')              # label = name of label

plt.ylabel('Deaths')

plt.title('Confirmed Deaths Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Confirmed.plot(kind = 'hist',bins =10,figsize = (8,8))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Confirmed.plot(kind = 'hist',bins = 10)

plt.clf()

# We cannot see plot due to clf()
# 1 - Filtering Pandas data frame

x = data['Deaths']>10000     

data[x]
# For pandas we can achieve index and value

for index,value in data[['Country/Region']][0:10].iterrows():

    print(index," : ",value)
# For example lets look frequency of Covid-19 types

print(data['Country/Region'].value_counts(dropna =False)) 

# For example max HP is 255 or min defense is 5

data.describe() #ignore null entries