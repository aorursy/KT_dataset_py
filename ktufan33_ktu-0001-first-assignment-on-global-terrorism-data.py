# This Python 3 environment comes with many helpful analytics libraries installed
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding = "cp1252")
data.head()
data.info()
data.describe()
data.corr()
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.country.plot(kind = 'line', color = 'g',label = 'region',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.iyear.plot(color = 'r',label = 'year',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='iyear', y='country',alpha = 0.5,color = 'red', figsize=(12,10))
plt.xlabel('year')              # label = name of label
plt.ylabel('country')
plt.title('year - country Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.iyear.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# Histogram
# bins = number of bar in figure
data.country.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
#create dictionary and look its keys and values
dictionary = {'mersin' : 'anamur','adana' : 'seyhan'}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['adana'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['adana']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)

# In order to run all code you need to take comment this line
del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
# 1 - Filtering Pandas data frame
x = data['country']>500     # There are only 3 pokemons who have higher defense value than 200
data[x]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
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
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['iyear']][0:3].iterrows():
    print(index," : ",value)
