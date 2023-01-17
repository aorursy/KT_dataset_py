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
data=pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.info()
#correlation map 

data.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
#line Plot

data.Speed.plot(kind="line",color='g',label='Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
#scatter plot

data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='red')

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title('Attack Defense Scatter Plot')
#Histogram Plot

data.Speed.plot(kind='hist',bins=50,figsize=(13,13))

plt.show()
dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
series = data['Defense']        # data['Defense'] = series

print(type(series))

data_frame = data[['Defense']]  # data[['Defense']] = data frame

print(type(data_frame))
x = data['Defense']>200 

data[x]
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
data[(data['Defense']>200) & (data['Attack']>100)]