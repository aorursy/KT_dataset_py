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
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.5f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.popularity.plot(kind = 'line', color = 'g',label = 'popularity',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.runtime.plot(color = 'r',label = 'runtime',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='popularity', y='runtime',alpha = 0.5,color = 'red')
plt.xlabel('popularity')              # label = name of label
plt.ylabel('runtime')
plt.title('popularity runtime Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.popularity.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.popularity.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
#create dictionary and look its keys and values
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())

# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/tmdb_5000_movies.csv')
series = data['popularity']        # data['Defense'] = series
print(type(series))
data_frame = data[['runtime']]  # data[['Defense']] = data frame
print(type(data_frame))
# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['popularity']>200     # There are only 3 pokemons who have higher defense value than 200
data[x]
# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
data[np.logical_and(data['popularity']>200, data['runtime']>100 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Defense']>200) & (data['Attack']>100)]