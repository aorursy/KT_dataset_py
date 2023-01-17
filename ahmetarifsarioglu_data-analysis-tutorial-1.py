# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import seaborn as sns #visualization 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# In this kernel a draft data analysis will be made using World Happiness Report 2017 data

data = pd.read_csv("../input/2017.csv")
data.columns
data.info()
data.head(10)
data.corr()
# correlation map

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)

plt.show()
# line plot

data.Family.plot(kind='line',color='b',label='Family', linewidth=1, alpha=0.5, 

grid=True, linestyle='-.')

data.Generosity.plot(color='r',label='Generosity', linewidth=1, alpha=0.5, 

grid=True, linestyle=':')

plt.legend(loc='upper right')    

plt.xlabel('x axis')

plt.ylabel('y label')

plt.title('Family-Generosity')

plt.show()

# Scatter Plot

# x= happiness score, y= family

data.plot(kind='scatter', x = 'Happiness.Score', y = 'Family', alpha = 0.3, color='red')

plt.xlabel('Happiness Score')

plt.ylabel('Family')

plt.title('Happiness-Family Scatter Plot')

plt.show()
# Histogram

data.Freedom.plot(kind='hist', bins=50, figsize=(12,12))

plt.show()
data.Freedom.plot(kind='hist', bins=30)

plt.clf()

#we cannot see the plot due to clf()
dictionary = {'spain':'madrid', 'usa':'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = 'barcelona'    # update existing entry

print(dictionary)

dictionary['france'] = 'paris'    # add new entry

print(dictionary)

del dictionary['spain']    # delete an entry

print(dictionary)

print('france' in dictionary)    # check if it is in dictionary or not

dictionary.clear()    # clear all entries in dictionaries

print(dictionary)
del dictionary

print(dictionary)    # gives error since dictionary exists no more
data = pd.read_csv("../input/2017.csv")
series = data['Freedom']    #pandas series

print(type(series))

data_frame = data[['Happiness.Score']]    # pandas Dataframe

print(type(data_frame))
# comparison operators

print(data.Family.head())

print(data.Freedom.head())

print(data.Freedom[1:2] > data.Family[1:2])

print(data['Family'][1:2] != data['Freedom'][1:2])

# Boolean Operators

print(True or False)

print(True and False)
# 1- Filtering Pandas DataFrame

x = data['Happiness.Score']>7.5    # only 3 countries with happiness score over 7.5 

data[x]
# 2- Filtering with logical_and

data[np.logical_and(data['Family']>1.5, data['Freedom']>0.6)]    #8 countries with over both values
# filtering with &

data[(data.Family > 1.5) & (data.Freedom > 0.6)]    # same result with the above code
# stay in loop if condition is true

i = 0

while i != 5:

    print('i: ', i)

    i += 1

print('i is equal to 5')
# stay in loop if condition is true

lis = [1,2,3,4,5]

for i in lis:

    print('i: ',i)

print('')



# enumerate index and value of list

for index,value in enumerate(lis):    # index-values in the list

    print('index ',index, ': ', value)

print('')



# We can use for loop to achive key and value of dictionary

dictionary = {'spain':'madrid', 'france':'paris'}

for key,value in dictionary.items():    # items() returns key-value pairs in dictionary

    print(key, ': ', value)



# For pandas we can achieve index and value

for index,value in data[['Happiness.Score']][0:2].iterrows():

    print(index, ': ', value)


