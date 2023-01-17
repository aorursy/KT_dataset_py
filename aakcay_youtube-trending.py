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
data = pd.read_csv('../input/CAvideos.csv')



data.info



data.columns
data.corr()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(6)
data.plot(kind='scatter', x='likes', y='dislikes',alpha = 0.5,color = 'red')

plt.xlabel('likes')              

plt.ylabel('dislikes')

plt.title(' Scatter Plot')     
data.likes.plot(kind = 'line', color = 'g',label = 'likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.dislikes.plot(color = 'r',label = 'dislikes',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right') 

plt.xlabel('likes')              

plt.ylabel('dislikes')

plt.title('Line Plot')           

plt.show()
data.likes.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
data.dislikes.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()