# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seaborn as sns
print(os.listdir("../input"))
data15=pd.read_csv('../input/2015.csv')
data16=pd.read_csv('../input/2016.csv')
data17=pd.read_csv('../input/2017.csv')
# Any results you write to the current directory are saved as output.
print(data15.info())
print(data16.info())
print(data17.info())
print(data15.columns)
print(data16.columns)
print(data17.columns)
print(data15.describe())
print(data16.describe())
print(data17.describe())
data15.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data15.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data16.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data16.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data17.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data17.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data15.head(50)
data16.head(50)
data17.head(100)

data15['Happiness Score'][0:10].plot(kind = 'line', color = 'g',label = 'Score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data15['Dystopia Residual'][0:10].plot(color = 'r',label = 'Distopia',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data15['Trust (Government Corruption)'].plot(kind='hist',bins=50 ,figsize=(12,12))
data15[np.logical_and(data15['Trust (Government Corruption)']>0.0, data15['Trust (Government Corruption)']<0.02 )] #numpy library logic
data16[np.logical_and(data16['Trust (Government Corruption)']>0.0, data16['Trust (Government Corruption)']<0.02 )] #numpy library logic
data17[np.logical_and(data17['Trust..Government.Corruption.']>0.0, data17['Trust..Government.Corruption.']<0.02 )] #numpy library logic
#Freedom Level Statistics




threshold = sum(data17.Freedom)/len(data17.Freedom)
print(threshold)
data17["Freedom_Level"] = ["high" if i > threshold else "low" for i in data17.Freedom]
data17.loc[:50,["Country","Freedom_Level","Freedom"]]
