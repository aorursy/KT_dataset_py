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
data = pd.read_csv("../input/tmdb_5000_movies.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.vote_average.plot(kind = 'line', color = 'g',label = 'runtime',linewidth=1,alpha = 1,grid = True,linestyle = ':')

data.revenue.plot(color = 'r',label = 'revenue',linewidth=1, alpha = 0.7,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('vote_average')              # label = name of label

plt.ylabel('Popularity')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

data.plot(kind='scatter', x='vote_average', y='runtime',alpha = 0.5,color = 'red')

plt.xlabel('Average Vote')              # label = name of label

plt.ylabel('Runtime')

plt.title('Vote-Runtime Scatter Plot')            # title = title of plot
data.runtime.plot(kind = 'hist',bins = 100,figsize = (12,12))

plt.show()
data.runtime.plot(kind = 'hist',bins = 50)

plt.clf()
dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = "barcelona"    # update existing entry

print(dictionary)

dictionary['france'] = "paris"       # Add new entry

print(dictionary)

del dictionary['spain']              # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
del dictionary
print(dictionary)
series = data['original_title']        

print(type(series))

data_frame = data[['original_title']]  

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
x = data['vote_average']>8.0     

data[x]
data[np.logical_and(data['vote_average']>8.0, data['original_language']=="it" )]
data[(data['vote_average']>8.0) & (data['original_language']=="it")]
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

for index,value in data[['id']][0:1].iterrows():

    print(index," : ",value)
