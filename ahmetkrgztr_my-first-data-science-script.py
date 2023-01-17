# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
data = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv')
data.info()

data.corr()
#correlation map
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(25)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.High.plot(kind = 'line', color = 'g',label = 'High',linewidth=1,alpha = 0.5,grid = True,linestyle = ':', figsize=(12,12))
data.Low.plot(color = 'r',label = 'Low',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.', figsize=(12,12))
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = high, y = low
data.plot(kind='scatter', x='High', y='Low',alpha = 0.5,color = 'red', figsize=(20,20))
plt.xlabel('High')              # label = name of label
plt.ylabel('Low')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.High.plot(kind = 'hist',bins = 500,figsize = (20,20))
plt.show()
#create dictionary and look its keys and values
dictionary = {'turkey' : 'istanbul','bulgaria' : 'sofia', 'germany' : 'berlin'}
print(dictionary.keys())
print(dictionary.values())

dictionary['turkey'] = "ankara"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['bulgaria']              # remove entry with key 'spain'
print(dictionary)
print('italy' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
series = data['High']        # data['Defense'] = series
print(type(series))
data_frame = data[['High']]  # data[['Defense']] = data frame
print(type(data_frame))
# Comparison operator
print(5 > 8)
print(2!=2)
# Boolean operators
print(False and True)
print(True or False)
x = data['High']>100     
data[x]


# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
data[np.logical_and(data['High']>200, data['Low']>100 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['High']>200) & (data['Low']>100)]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 7 :
    print('i is: ',i*2)
    i +=1 
print(i,' is equal to 7')
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5,'a','b','c','d','e']
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
for index,value in data[['High']][0:1].iterrows():
    print(index," : ",value)


