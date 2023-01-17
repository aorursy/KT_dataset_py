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
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.info()
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data1=data['price'].sort_values()

print(data1)
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.price.plot(kind = 'line', color = 'g',label = 'Price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.number_of_reviews.plot(color = 'r',label = 'number_of_reviews',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='price', y='number_of_reviews',alpha = 0.5,color = 'red')

plt.xlabel('price')              # label = name of label

plt.ylabel('number_of_reviews')

plt.title('price-number_of_reviews Scatter Plot')            # title = title of plot

plt.show()
data.price.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()