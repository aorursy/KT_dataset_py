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
data = pd.read_csv('../input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv')
data.info()
data.corr()
data.head(10)
f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.show()

# correlation map => There are few features which have close to the direct proportion.
# Line plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.priceRangeMin.plot(kind= 'line',color = 'm',label = 'Price Range Min',linewidth = 2,alpha=0.5,grid=True, linestyle = ':')

data.priceRangeMax.plot(kind ='line',color ='c',label = 'Price Range Max',linewidth = 2,alpha =0.5,grid=True,linestyle= '-')

plt.legend(loc='upper left') # puts label into upper left

plt.xlabel('x axis') # label name

plt.ylabel('y axis') # label name

plt.title('Line Plot') # title of plot

plt.show()

# Scatter plot

data.plot(kind='scatter',x='menus.amountMax',y='menus.amountMin',alpha=0.5,color='b')

plt.xlabel('Maximum number of menus')

plt.ylabel('Minimum number of menus')

plt.title('Amount of Maximum and Minimum Menus Scatter Plot')

plt.show()

plt.scatter(data.priceRangeMin,data.priceRangeMax,color='g',alpha=0.4)

plt.show()

# No label name,title and different features.
# Histogram

# bins = number of bar in figure

data.priceRangeMin.plot(kind='hist',bins=150,figsize=(11,11))

plt.show()
# clf() = cleans it up again you can start a fresh

data.priceRangeMin.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
dict = {"brand": "Ford","model": "Mustang","year": 1964} #define dictionary

print(dict.keys())

print(dict.values())

print(len(dict))

dict["year"] = 1970

print(dict)

dict.popitem() #removes the last inserted item ("year" :1970)

del dict["brand"] # delete brand key and ford value

print(dict)

dict.clear() #empties the dictionary

print(dict)
data.describe()
# Filtering Pandas Data Frame

filter1 = data["menus.amountMax"]>300

data[filter1]

filter2 = data.priceRangeMax <30

data[filter2]
## Filtering with logical_and

data[np.logical_and(data["menus.amountMax"]>300, data["priceRangeMax"]<=40 )]