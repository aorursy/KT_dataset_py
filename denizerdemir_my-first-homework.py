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
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/avocado.csv')
data.info()
data.corr()
f, ax = plt.subplots(figsize=(17,17))
sns.heatmap(data.corr(), annot=True, linewidths=.7, fmt='.3f',ax=ax)
plt.show()
data.head(15)
data.columns
data.AveragePrice.plot(kind = 'line', color = 'green',label = 'AveragePrice',linewidth=1,linestyle = ':')
data.year.plot(color = 'blue',label = 'year',linewidth=1,linestyle = '-.')
plt.legend(loc='upper right')    
plt.xlabel('x axis')            
plt.ylabel('y axis')
plt.title('Line Plot')          
plt.show()
data.plot(kind='scatter', x='AveragePrice', y='year',alpha = 0.5,color = 'green')
plt.xlabel('AveragePrice')              # label = name of label
plt.ylabel('year')
plt.title('AveragePrice year Scatter Plot')   
data.year.plot(kind = 'hist', bins = 13, figsize=(7,10))
plt.xlabel('Year') 
plt.title(' Year Histogram Plot') 
plt.show()
# clf() = cleans it up again you can start a fresh
data.year.plot(kind = 'hist',bins = 20)

# plt.clf()
# We cannot see plot due to clf()
dictionary = {'Turkey' : 'İstanbul','Suadi' : 'Makkah'}
print(dictionary.keys())
print(dictionary.values())
dictionary['Turkey'] = "İstanbul"    # update existing entry
print(dictionary)
dictionary['Germany'] = "Munih"       # Add new entry
print(dictionary)
del dictionary['Turkey']              # remove entry with key 'spain'
print(dictionary)
print('Germany' in dictionary)        # check include or not
# dictionary.clear()                   # remove all entries in dict
print(dictionary)
print(dictionary)
data = pd.read_csv('../input/avocado.csv')
series = data['year']        # data['Defense'] = series
print(type(series))
data_frame = data[['year']]  # data[['Defense']] = data frame
print(type(data_frame))
2==3 and 5==5
7<5
5!=3
5!=3 and 7<5
5!=3 or 7<5
data.head(100)
x = data['Large Bags']>400   
data[x]
data[np.logical_and(data['Large Bags']>1000, data['year']>700 )]
e = 0
while e != 10 :
    print('e is: ',e)
    e +=1 
print(e,' is equal to 10')
list = [3, 5, 8, 11, 13]
for i in list:
    print('i is: ',i)
print('...success')
for index, value in enumerate(list):
    print(index," = ",value)
print('...success')  
dictionary = {'İtalya':'Venedik','Almanya':'Berlin'}
for key,value in dictionary.items():
    print(key," = ",value)
print('...success')
for index,value in data[['Large Bags']][0:7].iterrows():
    print(index," = ",value)