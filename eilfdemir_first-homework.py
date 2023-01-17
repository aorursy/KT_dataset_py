# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")

data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize = (13, 13))
sns.heatmap(data.corr(), annot = True, linewidths =.5, fmt = '.1f', ax = ax)
plt.show()
data.head(10)
data.columns
# Matplotlib

# Line Plot
# label = label, color = color, linewidth = width of line, alpha = opacity, grid = grid, 
# linestyle = sytle of line

data.Timestamp.plot(kind = 'line', color = 'r', label = 'Timestamp', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')
data.Open.plot(color = 'g', label = 'Open', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc = 'upper right')  
plt.xlabel('x axis')            
plt.ylabel('y axis')
plt.title('Line Plot')            
plt.show()
# Scatter Plot 

data.plot(kind = 'scatter', x = 'Timestamp', y = 'Open', alpha = 0.5, color = 'red')
plt.xlabel('Timestamp')            
plt.ylabel('Open')
plt.title('Timestamp Open Scatter Plot')    
plt.show()
# Histogram
# bins = number of bar in figure

data.Weighted_Price.plot(kind = 'hist', bins = 50, figsize = (12,12))
plt.show()

# clf() = cleans it up again you can start a fresh

data.Weighted_Price.plot(kind = 'hist',bins = 50)
plt.clf()

# Dictionary

dictionary = {'turkey' : 'ankara','germany' : 'berlin'}
print(dictionary.keys())
print(dictionary.values())
dictionary['germany'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['japan'] = "tokyo"          # Add new entry
print(dictionary)
del dictionary['germany']              # remove entry with key 'spain'
print(dictionary)
print('japan' in dictionary)           # check include or not
dictionary.clear()                     # remove all entries in dict
print(dictionary)
print(dictionary) 
# Pandas

data = pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")
series = data['Low']       
print(type(series))
data_frame = data[['Low']] 
print(type(data_frame))
# Logic, control flow and filtering

print(5 > 7)
print(8 != 3)

# Booleans

print(True and False)
print(True or False)
x = data['Open'] > 2298847
data[x]
data[np.logical_and(data['Open'] > 2298847, data['Volume_(BTC)'] > 45.000000 )]
data[(data['Open'] > 2298847) & (data['Volume_(BTC)'] > 45.000000)]
# While and for loops

k = 0
while k != 7 :
    print('k is: ',k)
    k += 1 
print(k,' is equal to 7')
listt = [1,2,3,4,5]
for k in listt:
    print('k is: ',k)
print('')

# Enumerate index and value of list

for index, value in enumerate(listt):
    print(index," : ",value)
print('')   

# For dictionaries

dictionary = {'turkey':'ankara','japan':'tokyo'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas 
for index,value in data[['Open']][0:1].iterrows():
    print(index," : ",value)