# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns   # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/2015.csv')

data2 = pd.read_csv('../input/2016.csv')

data3 = pd.read_csv('../input/2017.csv')

# receive data / obtain data
print("Year 2015:")

data1.info()

print("\n Year 2016:")

data2.info()

print("\n Year 2017: ")

data3.info()

# we learn about the contents of the data
# Let's start processing 2017 data

data3.corr()
# correlation map

f,ax = plt.subplots(figsize = (21,21))

sns.heatmap(data3.corr(), annot= True, linewidths= .5, fmt= '.1f', ax= ax)

plt.show()

data3.head(24)

# Show 15 in our data from the beginning.
data3.columns

# What are the features of 2017 data?
# Line plot

data3.Freedom.plot(kind= 'line', color= 'red', label= 'Freedom', linewidth= 1, alpha = 0.8, grid= True, linestyle= ':')

data3.Generosity.plot(kind= 'line', color= 'blue', label= 'Generosity', linewidth= 1, alpha = 0.8, grid= True, linestyle= '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()

# Scatter Plot

# correlation between two variables

data3.plot(kind='scatter', x= 'Freedom', y= 'Generosity', alpha = 0.6, color= 'green')

plt.xlabel('Freedom')

plt.xlabel('Generosity')

plt.title('Freedom-Generosity Scatter Plot')

# Histogram

# Let's learn about the frequency of feature

data3.Freedom.plot(kind= 'hist', bins= 65, figsize= (21,21))

plt.show()

# bins = number of bar in figure
# clf() = cleans it up again you can start a fresh

data3.Generosity.plot(kind= 'hist', bins =65)

plt.clf()

# we can't see plot due to clf()

data3 = pd.read_csv('../input/2017.csv')

series = data3['Freedom']

print(type(series))

data_frame = data3[['Freedom']]

print(type(data_frame))
# With Pandas, let's make a dataframe filtering example

filter1 = data3['Economy..GDP.per.Capita.'] > 1.5

data3[filter1]

# we search for something more specific

data3[(data3['Happiness.Rank'] < 36) & (data3['Economy..GDP.per.Capita.'] > 1.6)]

# Are these the most livable countries?