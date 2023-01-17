# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from subprocess import check_output

# Any results you write to the current directory are saved as output.
#Importing data

data = pd.read_csv('../input/master.csv')

data.info()
#Checking the correlation

data.corr()
#Generating the correlation map with the help of sns.heatmap

f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data.corr(), annot=True, linewidths = 1, fmt = '.2f',ax = ax) #using the seaborn and heatmap

plt.show() #matplotlib.pyplot library
#Checkng the whole data with rows and columns

data.head(12)#The number determines how many rows will be shown
#To get an information about all the columns 

data.columns
#Scatter pot

#x = population and y = suicide number

data.plot(kind = 'scatter', x = 'population', y = 'suicides_no', alpha = 0.5, color = 'b' )

plt.xlabel('Population')

plt.ylabel('Suicide number')

plt.title(' Population vs. Suicide number')

plt.show()
# Histogram

# bins = number of bar in figure

data.suicides_no.plot(kind = 'hist', bins = 50, figsize = (12,12))

plt.show()
# 1 - Filtering Pandas data frame

x = data['suicides_no']>20

data[x]
plt.hist(data['year'], color = 'blue', edgecolor = 'black',bins = 50)

plt.xlabel('year')

plt.ylabel('suicides_no')

plt.show()
#Line plot



data.suicides_no.plot(kind = 'line', color = 'g',label = 'suicides_no',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.population.plot(color = 'r',label = 'population',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()