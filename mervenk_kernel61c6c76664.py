# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/advertising/advertising.csv")

data.info()
data.corr()
f,ax=plt.subplots(figsize=(16,16))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
data.columns


data.Age.plot(color = 'r',label = 'Age',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='Area Income', y='Daily Internet Usage',alpha = 0.5,color = 'red')

plt.xlabel('Area Income')              # label = name of label

plt.ylabel('Daily Internet Usage')

plt.title('Area Income Daily Internet Usage Scatter Plot')            # title = title of plot

plt.show()
data.Age.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
data.Age.plot(kind = 'hist',bins = 50)

plt.clf()
x = data['Age']>50     

print(x)

data[x]
data[np.logical_and(data['Age']>50, data['Country']=='Bulgaria' )]