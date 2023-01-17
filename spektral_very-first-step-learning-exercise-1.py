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
print (os.listdir("../input"))


from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2016.csv')
data.info()
data.corr()
# correlation map
f,ax = plt.subplots(figsize = (12,12))
sns.heatmap(data.corr(), annot= True, linewidths=1, fmt='.1f',ax=ax)
plt.show()
data.head(25)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Freedom.plot(kind = 'line', color = 'g',label = 'Freedom',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Generosity.plot(color = 'r',label = 'Generosity',linewidth=1, alpha = 0.5,grid = True, linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

data.plot(kind='scatter', x='Happiness Rank', y='Freedom',alpha = 0.5,color = 'red')
plt.xlabel('Happiness Rank')              # label = name of label
plt.ylabel('Freedom')
plt.title('Attack Defense Scatter Plot')
data.Freedom.plot(kind = 'hist',bins = 50,figsize = (24,24))
plt.show()
# clf() = cleans it up again you can start a fresh
data.Freedom.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
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
print (dictionary)
data = pd.read_csv('../input/2016.csv')
series = data['Economy (GDP per Capita)']
print(type(series))
data_frame = data[['Happiness Rank']]
print(type(data_frame))
print(78 > 244)
print(3!=2)
print(0.9 > 6)
# Boolean operators
print(True and False)
print(True or False)
print(False and False)
a = data['Happiness Score'] > 6.9
data[a]
data[np.logical_and(data['Freedom']>0.5, data['Family']>1.1 )]
data[(data['Freedom']>0.5) & (data['Family']>1.1 )]
i = 3
while i != 12 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 12')
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
for index,value in data[['Health (Life Expectancy)']][0:1].iterrows():
    print(index," : ",value)