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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/creditcard.csv')
data.info
data.head (10)
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(35, 35))
sns.heatmap(data.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax)
plt.show()
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.V2.plot(kind = 'line', color = 'r',label = 'V2',linewidth=2,alpha = 0.8,grid = True,linestyle = ':')
data.V4.plot(color = 'y',label = 'V4',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = V2, y = V4
data.plot(kind='scatter', x='V2', y='V4',alpha = 0.8,color = 'purple')
plt.xlabel('V2')              # label = name of label
plt.ylabel('V4')
plt.title('V2-V4 Scatter Plot')            # title = title of plot
plt.show()
# Histogram
data.V10.plot(kind = 'hist',bins = 25,figsize = (8,8))
plt.show()
#create dictionary and look its keys and values
dictionary = {'istanbul' : 'kadıköy','ankara' : 'çankaya', 'yozgat' : 'boğazlıyan'}
print(dictionary.keys())
print(dictionary.values())
dictionary['istanbul'] = "kadıköy"    # update existing entry
print(dictionary)
dictionary['ankara'] = "çankaya"       # Add new entry
print(dictionary)
del dictionary['ankara']              # remove entry with key 'spain'
print(dictionary)
print('istanbul' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)

print(dictionary)
data.info
series = data['V4']      
print(type(series))
data_frame = data[['V4']] 
print(type(data_frame))
# Comparison operator
print(5 > 4)
print(4 > 5)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['V2']>20   # There are only 2 V2 who have lower V1,V3,V7
data[x]
# 2 - Filtering pandas with logical_and
# # There are only 2 credit car who have higher V2 value than 20 and lower V4 than value 10
data[np.logical_and(data['V2']>20, data['V4']<12)]
data[(data['V2']>20) & (data['V4']<12)]
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
dictionary = {'istanbul' : 'kadıköy','ankara' : 'çankaya', 'yozgat' : 'boğazlıyan'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['V2']][1:2].iterrows():
    print(index," : ",value)

