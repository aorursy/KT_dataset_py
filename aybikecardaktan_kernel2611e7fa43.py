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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/bitmex_xbtusd_1m_2016-12-31_2018-06-17.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.head(21)
data.columns
data.volume.plot(kind = 'line', color = 'red',label = 'volume',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.open.plot(color = 'green',label = 'open',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper left')     # legend = puts label into plot

plt.xlabel('open axis')              # label = name of label

plt.ylabel('volume axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='volume', y='open',alpha = 0.5,color = 'green')

plt.xlabel('volume')              # label = name of label

plt.ylabel('open')

plt.title('Volume_Open Scatter Plot')            # title = title of plot

plt.show()
data.open.plot(kind = 'hist',bins = 60,figsize = (12,12),color='pink')

plt.show()
dictionary = {'open' : 'bitcoin','volume' : 'money'}

print(dictionary.keys())

print(dictionary.values())
dictionary['open'] = "crypto"    # update existing entry

print(dictionary)

dictionary['close'] = "bitcoin"       # Add new entry

print(dictionary)

del dictionary['open']              # remove entry with key 'spain'

print(dictionary)

print('volume' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
print(dictionary) 
data = pd.read_csv('../input/bitmex_xbtusd_1m_2016-12-31_2018-06-17.csv')
series = data['open']        # data['open'] = series

print(type(series))

data_frame = data[['open']]  # data[['open']] = data frame

print(type(data_frame))
# Comparison operator

print(6 > 8)

print(4!=5)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['open']>999

data[x]
data[np.logical_and(data['open']>999, data['close']>999 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['open']>999) & (data['close']>999)]
i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
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

dictionary = {'open' : 'bitcoin','volume' : 'money'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['open']][0:1].iterrows():

    print(index," : ",value)