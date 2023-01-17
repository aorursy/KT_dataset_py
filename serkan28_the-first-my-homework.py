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
data = pd.read_csv('../input/creditcard.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot = True,linewidths = 0.5,fmt = '.1f', ax = ax)
data.head()
data.columns
data.V3.plot(kind = 'line', color = 'g',label = 'V3',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.V6.plot(color = 'r',label = 'V6',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='V1', y='V18',alpha = 0.5,color = 'red')
plt.xlabel('V1')              # label = name of label
plt.ylabel('V18')
plt.title('V1 V18 Scatter Plot') 
data.V3.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data.V3.plot(kind = 'hist',bins = 50)
plt.clf()
dictionary = {'Serkan':'KIRCA','Ekol':'Lojistik'}
keys = dictionary.keys()
values = dictionary.values()
items = dictionary.items()
print(keys)
print(values)
print(items)
dictionary['Ekol'] = 'RO-RO'
print(dictionary)
dictionary['Mekatronik'] = 'Mühendisi'
print(dictionary)
del dictionary['Ekol']
print(dictionary)
print('Serkan' in dictionary)
dictionary.clear()
print(dictionary)

data = pd.read_csv('../input/creditcard.csv')
data.columns
series = data['V12']
print(type(series))
data_frame = data[['V12']]
print(type(data_frame))
x = data['V3'] >4
data[x] # We are writing only true value here.
data[np.logical_and(data['V3']>4,data['V12'] >0)]
x = (data['V3'] > 4) & (data['V12'] > 0)
data[x]
data[(data['V3'] > 4) & (data['V12'] > 0)]
i = 0
while i!= 5:
    print("i is:",i)
    i +=1
print(i, "is equal to 5")
liste = [1,2,3,4,5,6,7]
for i in liste:
    print("i is :",i)
print("We written all of liste...")
lis = [1,2,3,4,5]
print('in:val')
for index,value in enumerate(lis): # We can use enumarate , if we write in list element value and index 
    print(index,":",value)
print("The list value was written with index by us")

dictionary = {'Serkan':'KIRCA','Mekatronik':'Mühendisi'}
for key,value in dictionary.items():
    print(key,':',value)
    
print("Thats all :)")
# we want to ashive to pandas
for index,value in data[['V3']][0:3].iterrows():
    print(index,':',value)
a = data[['V3']][0:5]
print(a)