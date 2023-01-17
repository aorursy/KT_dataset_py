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
data = pd.read_csv('../input/data.csv')
data.info()
data.head(10)
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.Finishing.plot(kind = 'line', color = 'g',label = 'Finishing',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Age.plot(color = 'r',label = 'Age',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='Finishing', y='Age',alpha = 0.5,color = 'red')
plt.xlabel('Finishing')              
plt.ylabel('Age')
plt.title('Finishing Age Scatter Plot')  
data.SprintSpeed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
x = data['Finishing']>90     # There are only 3 pokemons who have higher defense value than 200
data[x]
data[np.logical_and(data['Finishing']>90, data['Age']<31 )]
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1
print(i,' is equal to 5')

