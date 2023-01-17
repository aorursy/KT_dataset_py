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
data= pd.read_csv("../input/2015.csv")

data1= pd.read_csv("../input/2016.csv")

data2= pd.read_csv("../input/2017.csv")
data.info()  #general info about data

data1.info()

data2.info()
data2.describe() #statistical info about data2
data2.corr()   #correlation of parameters
#correlation map

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data2.corr(), annot=True, linewidths=.7, fmt= '.2f',ax=ax)

plt.show()
data2.head(5) #first 5 happiest countries 
data2.tail(5) #last 5 unhappiest countries 
data2.columns #columns of data2
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data2.Freedom.plot(kind = 'line', color = 'blue',label = 'Freedom',linewidth=1,alpha = 3,grid = True,linestyle = ':')

data2.Generosity.plot(color = 'pink',label = 'Generosity',linewidth=1, alpha = 1,grid = True,linestyle = '-.')

plt.legend(loc='upper left')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Economy..GDP.per.Capita, y = Health..Life.Expectancy

data2.plot(kind='scatter', x='Economy..GDP.per.Capita.', y='Health..Life.Expectancy.',alpha = 1,color = 'purple')

plt.xlabel('Economy..GDP.per.Capita.')              # label = name of label

plt.ylabel('Health..Life.Expectancy.')

plt.title('GDP vs Life Expectancy Scatter Plot')            # title = title of plot

plt.show ()
# Histogram

# bins = number of bar in figure

data2.Freedom.plot(kind = 'hist',bins = 20,figsize = (9,9), color='orange')

plt.show()
x=data2['Happiness.Score'] < 5

data2[x]
data2[np.logical_and(data2['Happiness.Score']>5,data2['Freedom']>0.5)]