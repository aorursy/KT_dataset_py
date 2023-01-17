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

print(os.listdir("../input/heart-disease-uci/"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(16)
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.age.plot(kind = 'line', color = 'g',label = 'age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.cp.plot(color = 'r',label = 'cp',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='Scatter', x='age', y='cp',alpha = 0.5,color = 'red')

plt.xlabel('age')              # label = name of label

plt.ylabel('cp')

plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.age.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()