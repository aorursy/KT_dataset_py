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



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/USvideos.csv')
data.info() # preview of dataset
data.corr()
# correlation map

f,ax = plt.subplots(figsize = (15, 15))

sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt= '.1f', ax=ax)

plt.show()
data.head(15)
data.columns # shows features
# Line plot



data.likes.plot(kind = 'line', color = 'g', label = 'Likes', linewidth = 1.2, alpha = 0.5, grid = True, linestyle = ':')

data.dislikes.plot(color = 'r', label = 'Dislikes',linewidth = 1.2,alpha = 0.5 , grid = True, linestyle = '-.')

plt.legend(loc='upper left')

plt.xlabel('x- axis')

plt.ylabel('y- axis')

plt.title('Line Plot')

plt.show()
# Scatter plot

# x = likes, y = views

data.plot(kind = 'scatter', x = 'likes', y = 'views',alpha = 0.5, color = 'blue')

plt.xlabel('Likes')

plt.ylabel('Views')

plt.title('Likes Views Scatter Plot')

plt.show()
# Histogram

data.comment_count.plot(kind = 'hist', bins = 10, figsize = (11,11))

plt.show()
data.comment_count.plot(kind = 'hist', bins = 10,)

plt.clf()

# dictionart basics



dictionary = {'Forest' : 'Wolf', 'Sea' : 'Dolphin'}

print(dictionary.keys())

print(dictionary.values())



dictionary['Forest'] = "Tiger" # update existing entry

print(dictionary)

dictionary['Desert'] = "Snake" # add new entry

print(dictionary)

del dictionary['Forest'] # remove entry with key 'Forest'

print(dictionary)

print('bla' in dictionary) # check 'bla' is exist or not

dictionary.clear() # remove all entries in dictionary

print(dictionary)

# del dictionary # delete entire dictionary

# 1.Filtering Pandas Data Frame

x = data['dislikes'] > 1000000  # There are twelve videos have dislikes over 1.000.000

data[x]
# 2.Filtering pandas with logical_and

# There are twelve videos which have dislikes and likes over 1.000.000

data[np.logical_and(data['likes']>1000000, data['dislikes']>1000000)]



# it is also same with previous one,

# data[(data['likes']>1000000) & (data['dislikes']>1000000)]
# Stay in loop if condition( i is not equal 5) is true

lis = [10,20,30,40,50]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:10, 1:20, 2:30, 3:40, 4:50

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary with items()

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value with iterrows()

for index,value in data[['dislikes']][0:1].iterrows():

    print(index," : ",value)

avgView = sum(data.views) / len(data.views)

data['AboveAverage'] = ["True" if i > avgView else "False" for i in data.views]

data.loc[:10,["AboveAverage","views"]]