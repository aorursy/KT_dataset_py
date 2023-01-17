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
data = pd.read_csv('../input/oasis_cross-sectional.csv')
data.info
data.corr()

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
#Matplotlib
#data.Age.plot(kind = 'line', color = 'r',label='Age',linewidth='1',alpha=0.5,grid=True,linestyle=':')
#data.M/F.plot(kind='line',color='g',label='M/F',linewidth='1',alpha=0.5,grid=True,linestyle='-.')
data.eTIV.plot(kind = 'line', color = 'g',label = 'eTIV',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.nWBV.plot(color = 'r',label = 'nWBV',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# x = Age, y = eTIV
data.plot(kind='scatter', x='Age', y='eTIV',alpha = 0.5,color = 'red')
plt.xlabel('Age')
plt.ylabel('eTIV')
plt.title('Age eTIV Scatter Plot')   
data.Age.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
dictionary = {'MRI1' : 'Female','MRI2' : 'Female','MRI3' : 'male','MRI4' : 'male','MRI5' : 'Female'}
print(dictionary.keys())
print(dictionary.values())
print(dictionary)
dictionary['MRI1'] = "MALE"    # update existing entry
print(dictionary)
dictionary['MRI6'] = "female"       # Add new entry
print(dictionary)
del dictionary['MRI3']              # remove entry with key 'spain'
print(dictionary)
print('MRI6' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict
print(dictionary)
series = data['Age']       
print(type(series))
data_frame = data[['Age']]  
print(type(data_frame))
# 1 - Filtering Pandas data frame
x = data['Age']>45     
data[x]
# 2 - Filtering pandas with logical_and
data[np.logical_and(data['Age']>45, data['Educ']>2.0 )]

data[(data['Age']>50) & (data['Educ']>2.0)]
