# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #vizulation tools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2016.csv')
data.info()
data.corr()
data.head(10)
#Correlation map
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt='0.1f', ax=ax)
plt.show()
data.columns
#Line plot (Family-Freedom )
#Red:Family
#Green:Freedom
data.Family.plot(kind='line', color='r', label='Family', linewidth=1, alpha=0.9, grid=True, linestyle=':')
data.Freedom.plot(kind='line', color='g', label='Freedom', linewidth=1, alpha=0.9, grid=True, linestyle='-.')
#plt.leged(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

#Scatter Plot (Freedom - Happiness Score)
data.plot(kind='scatter', x='Freedom', y='Happiness Score', alpha=0.5, color='red')
plt.xlabel('Freedom')
plt.ylabel('Happiness Score')
plt.title('Freedom Happiness Score Scatter Plot')

#Scatter Plot (Economy - Happiness Score)
data.plot(kind='scatter', x='Economy (GDP per Capita)', y='Happiness Score', alpha=0.5, color='red')
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title('Economy Happiness Score Scatter Plot')

#Histogram (Freedom Frequency)
data.Freedom.plot(kind='hist', bins=50, figsize=(15,15))
plt.show()
data.Freedom.plot(kind='hist', bins=50, figsize=(15,15))
plt.clf()
# 1 - Filtering Pandas data frame
#There are only 12 country which have higher happiness score than 7 and higher freedom value than 0.5. 
x = data['Happiness Score']>7
data[x]
# 2 - Filtering pandas with logical_and
data[np.logical_and(data['Happiness Score']>7, data['Freedom']>0.5)]

#There are only 12 country which have higher happiness score than 7 and higher freedom value than 0.5. 
data[(data['Happiness Score']>7) & (data['Freedom']>0.5)]
