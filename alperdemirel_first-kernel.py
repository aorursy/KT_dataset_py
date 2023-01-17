# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualizations

import seaborn as sns # data visualizations



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading the dataset

data=pd.read_csv('../input/world-happiness/2017.csv')

data.shape
#information the data

data.info()
#describing the data

data.describe()
#checking if there are any null values

data.isnull().sum()
#checking the head of the data

data.head(10)
#checking the tail of the data

data.tail(10)
#data features

data.columns
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Freedom.plot(kind = 'line', color = 'red',label = 'Freedom',linewidth=1,alpha = 1.0,grid = True,linestyle = ':')

data.Generosity.plot(color = 'blue',label = 'Generosity',linewidth=1, alpha = 1.0,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.show()
#x = freedom, y = generosity

data.plot(kind='scatter', x='Freedom', y='Generosity',alpha = 0.5,color = 'black')

plt.xlabel('Freedom')           # label = name of label

plt.ylabel('Generosity')

plt.show()
data[np.logical_and(data['Freedom']>0.5, data['Generosity']>0.5 )]