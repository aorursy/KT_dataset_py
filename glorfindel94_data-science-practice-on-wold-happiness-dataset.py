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
data = pd.read_csv('../input/2017.csv')

data.info()

data.head(10)

data = data.rename(columns={'Happiness.Score': 'Happiness'})

data = data.rename(columns={'Economy..GDP.per.Capita.': 'GDP'})
data.corr()

#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns

# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Happiness.plot(kind = 'line', color = 'g',label = 'Happiness',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.GDP.plot(color = 'r',label = 'GDP',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x ')              # label = name of label

plt.ylabel('y ')

plt.title('Happiness and GDP')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Happiness, y = GDP

data.plot(kind='scatter', x='Happiness', y='GDP',alpha = 0.5,color = 'red') # Happiness vs GDP scatter plot

plt.xlabel('Happiness')              # label = name of label

plt.ylabel('GDP')

plt.title('Happiness vs GDP')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.Happiness.plot(kind = 'hist',bins = 50,figsize = (12,12)) # happiness histogram

plt.show()
#create dictionary and look its keys and values

dictionary = {'Iceland' : ' rank 3','Norway' : 'rank 2'} # happiness ranks of iceland and norway in our dictionary

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['Norway'] = "rank 1"    # update existing entry

print(dictionary)

dictionary['Denmark'] = " rank 2 "       # Add new entry

print(dictionary)

#del dictionary['Denmark']              # remove entry with key 'Denmark'

#print(dictionary)

print('Denmark' in dictionary)        # check include or not

#dictionary.clear()                   # remove all entries in dict

#print(dictionary)
series = data['GDP']        # data['GDP'] = series

print(type(series))

data_frame = data[['GDP']] #list  # data[['GDP']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['GDP']>1.5     # There are 12 countries that have higher than 1.5 GDP value 

data[x]

# 2 - Filtering pandas with logical_and

# There are 6 countries that have higher gdp value than 1.5 and family value than 1.4

data[np.logical_and(data['GDP']>1.5, data['Family']>1.4 )]

# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['GDP']>1.5) & (data['Family']>1.4)]



# Stay in loop if condition

lis = ['Norway rank 1','Denmark rank 2','Iceland rank 3']

for i in lis:

    print(' Happines Rank = ',i)

print('')
# Enumerate index and value of list

lis1 = ['Norway ','Denmark ','Iceland']

for index, value in enumerate(lis1):

    print(" Rank=",index,' is ',value)

print('')   
# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Iceland': ' rank 3', 'Norway': 'rank 1', 'Denmark': ' rank 2 '}



for key,value in dictionary.items():

    print(key," is ",value)

print('')
# For pandas we can achieve index and value

for index,value in data[['GDP']][-1:].iterrows(): # lowest value of gdp 

    print(index," : ",value)