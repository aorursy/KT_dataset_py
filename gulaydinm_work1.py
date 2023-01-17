# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting 
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/AADF-data-major-roads.csv')
data.info()
data.describe()
data.corr() #gives us the correlation in whole dataset.That's how we'll be able to see relation between each data..
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=False, linewidths=.10, fmt='.1f', ax=ax)
plt.show()

#more detailed plot
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True , linewidths=.10 ,fmt='.1f', ax=ax)
plt.show()
data.head(10)
data.columns
#Following codes are written for a simple line plot..
data.FdPC.plot(kind= 'line' , color='green' ,label = 'FdPC' ,linewidth=3, alpha=0.5 ,grid =True ,linestyle=':')
data.LenNet.plot(kind='line' , color='red' ,label ='LenNet' , linewidth=3, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
#scatter plot is good for correlated variables.
data.plot(kind='scatter' ,x='FdHGVR3' ,y='FdHGVR4' ,alpha=0.5 ,color='green',figsize=(10,10))
plt.xlabel('FdHGVR3')
plt.ylabel('FdHGVR4')
plt.title('Scatter Plot for FdHGVR3 and FdHGVR4')
plt.show()
#histogram plot for distribution of numeric data.
data.FdCar.plot(kind='hist', bins =50,figsize=(10,10))
plt.show()
data.FdBUS.plot(kind='hist', bins=25, figsize=(10,10))
plt.show()
dictionary ={"South West" : "Isles of Scilly" ,"Wales" : "Swansea"}
print(dictionary.keys())
print(dictionary.values())
dictionary['Wales'] = "Newport" #update dictionary
print(dictionary)
#del dictionary['Wales']        #deletes a key in our dictionary
#dictionary.clear()             #deletes all entries in our dictionary
print('Swansea' in dictionary)
series = data['FdCar']
print(type(series))
#print(series)
data_frame = data[['FdCar']]
print(type(data_frame))
#comparision operators via print function
print(True or False)
print(True and False)
#filtering data
filtered_data = data['FdCar'] < 1000
data[filtered_data]
data[np.logical_and(data['FdCar']<1000,data['FdBUS'] < 20)]
data[(data['FdCar']<1000) & (data['FdBUS'] < 20)] #It will return same result as previous one
