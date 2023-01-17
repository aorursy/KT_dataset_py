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
data = pd.read_csv("/kaggle/input/steam-store-games/steam.csv")

data.info()
data.corr()
f,ax = plt.subplots(figsize=(16,16))

sns.heatmap(data.corr(), annot=True, linewidth=5, fmt= ".1f", ax = ax)

plt.show()
data.head(5)
data.columns
data.positive_ratings.plot(kind = 'line', color = 'y',label = 'positive_ratings',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.negative_ratings.plot(color = 'g',label = 'negative_ratings',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

data.average_playtime.plot(color = 'r',label = 'average_playtime',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

data.price.plot(color = 'g',label = 'price',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
data.plot(kind='scatter', x='price', y='average_playtime',alpha = 0.5,color = 'red')

plt.xlabel('price')              

plt.ylabel('average_playtime')

plt.title('Price Average Playtime Scatter Plot')            

plt.show()
data.price.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
data.price.plot(kind = 'hist',bins = 50)

plt.clf()
x = data["price"] > 120

print(x)

data[x]
data[np.logical_and(data['price']>100, data['average_playtime']>1000 )]