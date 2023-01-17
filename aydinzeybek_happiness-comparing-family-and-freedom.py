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
data = pd.read_csv('../input/2017.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(), annot=True, linewidths=1, fmt= '.2f', ax=ax)

plt.show()
data.head(15)
data.columns
data.Family.plot(kind = 'line', color = 'g',label = 'Family',linewidth=2,alpha = 0.9,grid = True,linestyle = ':' ,figsize=(15,15))

data.Freedom.plot(color = 'r',label = 'Freedom',linewidth=2, alpha = 0.9,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
data.plot(kind='scatter', x='Family', y='Freedom',alpha = 0.4,color = 'red', figsize=(4,4))

plt.xlabel('Family')              

plt.ylabel('Freedom')

plt.title('Family Freedom Scatter Plot')     

plt.show()
data.Freedom.plot(kind = 'hist',bins = 70,figsize = (10,10), color = 'purple')

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Histogram') 

plt.show()