# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
### We first read the data to a dataframe and see column names, first and last 10 rows

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.info()

print("First 10 lines: \n")

data.head()

data.tail()
# Then we check if there is any correlaction between the columns

data.corr()

#and visualize it in a heat map

#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(18, 18))

data.price.plot(kind = 'line', color = 'g',label = 'Price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',ax=ax)

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='price', y='availability_365',alpha = 0.5,color = 'red')

plt.xlabel('Price')              # label = name of label

plt.ylabel('availability_365')

plt.title('Price -availability_365 Scatter Plot')    

plt.show()
# Histogram

# bins = number of bar in figure

data.availability_365.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()