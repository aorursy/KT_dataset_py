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
data = pd.read_csv('../input/data.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.ShotPower.plot(kind = 'line', color = 'g',label = 'Shot Power',linewidth=3,alpha = 0.5,grid = True,linestyle = ':')

data.LongShots.plot(figsize=(10,10), color = 'r',label = 'Long Shots',linewidth=3, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='ShotPower', y='LongShots',alpha = 0.1,color = 'red')

plt.xlabel('Shots Power')              # label = name of label

plt.ylabel('Long Shots')

plt.title('Attack Defense Scatter Plot')            # title = title of plot
data.ShotPower.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.ShotPower.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
# 1 - Filtering Pandas data frame

x = data['Strength']>96    # He is a wrestler :)

data[x]
# 2 - Filtering pandas with logical_and

# This kid does a lot of work. :)

data[np.logical_and(data['Age']<=18, data['Dribbling']>85 )]
data = pd.read_csv('../input/data.csv')

data.head()
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
data.shape
data.info()
# For example let's look frequency of players nationality

print(data['Nationality'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 303 Turkish players or 1 New Caledonia's player.
# For example max Jmuping is 95 or min ShotPower is 2 :)))

# Brother, I play you don't bother. :)

# Jumping player must played basketball.

data.describe() #ignore null entries
data.boxplot(column='Age',by = 'Work Rate')

plt.show()
new_data = data.head()    # I only take 5 rows into new data

new_data
melted = pd.melt(frame=new_data,id_vars = 'Name', value_vars= ['Dribbling','SprintSpeed'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['BallControl'].head()

data2= data['LongPassing'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col