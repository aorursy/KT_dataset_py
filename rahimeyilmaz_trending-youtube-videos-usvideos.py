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
data = pd.read_csv('../input/youtube-new/USvideos.csv')
data.info()
#correlation map

f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot = True,linewidths=.5, fmt = '.1f' , ax=ax ,cmap="YlGnBu")
data.head(1)
#line plot

data.views.plot(kind = 'line', color='r', label = 'Views', linewidth = 1, alpha=0.5,grid=True,linestyle = ':' )

data.comment_count.plot(color='g', label = 'Comment Count', linewidth = 1, alpha=0.5,grid=True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel=('x axis')

plt.ylabel=('y axis')

plt.title=('Line Plot')
#Scatter Plot

#x= Views  y= Comment Count

data.plot(kind='scatter' , x='views', y='comment_count', alpha=0.5, color ='red')

plt.xlabel=('views')

plt.ylabel=('comment_count')

plt.title=('Views - Comment Count Scatter Plot')

#Histogram

#bins = number of bar in figure

data.views.plot(kind='hist', bins=20, figsize=(5,5), color='purple')

plt.show()

#Filtering Pandas Data Frame

x=data['views'] > 1000000

data[x]
#Filtering Pandas with logical_and

data[(data['views']>1000000) & (data['comment_count']>1000) ]
#for pandas we can achieve index and value

for index,value in data[['title']][0:10].iterrows():

    print(index, " : " , value)