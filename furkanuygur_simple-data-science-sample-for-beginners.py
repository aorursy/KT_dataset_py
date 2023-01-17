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
data = pd.read_csv('../input/videogamesales/vgsales.csv')
#information of data

data.info()
#the ratio between data

data.corr() 
#top 10 data but ----- data.head() = top 5 data

data.head(10)
#last 10 data

data.tail(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.NA_Sales.plot(kind = 'line', color = 'r',label = 'North America Sales',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize = (15,15))

data.EU_Sales.plot(kind = "line",color = 'b',label = 'European Sales',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.',figsize = (15,15))

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot =  correlation between two variables

# x = North America Sales , y = European Sales

data.plot(kind='scatter', x='NA_Sales', y='EU_Sales',alpha = 0.5,color = 'red',figsize = (12,12))

plt.xlabel('North America Sales')              # label = name of label

plt.ylabel('European Sales')

plt.title('Sales Scatter Plot')            # title = title of plot

plt.show()
#Alternative Code

plt.scatter(data.NA_Sales,data.EU_Sales,color = "red",alpha = 0.5) 
# Histogram

# bins = number of bar in figure

data.Year.plot(kind = 'hist',bins = 50,figsize = (14,14),label = 'Year of release')

plt.legend()

plt.show()
# clf() = cleans it up again you can start a fresh

data.Year.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dic = {'game' : 'The Legend of Zelda','Year' : '1991'}

print(dic.keys())

print(dic.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dic['game'] = "Super Mario"    # update existing entry

print(dic)

dic['Genre'] = "Action"       # Add new entry

print(dic)

del dic['game']              # remove entry with key 'spain'

print(dic)

print('Genre' in dic)        # check include or not

#dic.clear()                   # remove all entries in dict.Delete entire dictionary from memories so it gives error because dictionary is deleted

print(dic)
data = pd.read_csv('../input/videogamesales/vgsales.csv')
# 1 - Filtering Pandas data frame

x = data['Year']>1999     # How many games are there after 1999?

data[x]
# 2 - Filtering pandas - use '&' for filtering.

data[(data['Year']>1999) & (data['Global_Sales']<20)]

# Stay in loop if condition( i is not equal 5) is true

i = 0

while i < 5 :

    print('i is: ',i)

    i = i + 1

print(i,' is not less than 5')
# Stay in loop if condition( i is not equal 5) is true

lis = [1999,2000,2001,2002,2003]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'game':'The Legend of Zelda','Year':'1991'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Year']][0:3].iterrows():

    print(index," : ",value)
