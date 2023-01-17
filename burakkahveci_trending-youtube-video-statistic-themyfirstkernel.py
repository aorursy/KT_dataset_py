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
data = pd.read_csv('../input/USvideos.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.dislikes.plot(kind = 'line', color = 'g',label = 'Dislikes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.comment_count.plot(color = 'r',label = 'Comment Count',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='dislikes', y='comment_count',alpha = 0.5,color = 'red')

plt.xlabel('Dislake')              # label = name of label

plt.ylabel('Comment_Count')

plt.title('Dislike-CommenCount Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.views.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
#create dictionary and look its keys and values

dictionary = {'turkey' : 'ankara','Italy' : 'roma'}

print(dictionary.keys())

print(dictionary.values())
dictionary['turkey'] = "konya"

print(dictionary)

dictionary['Italy'] = "napoli"       

print(dictionary)

del dictionary['turkey']              

print(dictionary)

print('Italy' in dictionary)     

dictionary.clear()                  

print(dictionary)
series = data['views']

print(type(series))

data_frame = data[['views']]

print(type(data_frame))

x = data['views']>2000000     

data[x]
data[np.logical_and(data['views']>2000000, data['likes']>100000 )]