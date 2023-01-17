# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/pokemon-challenge/pokemon.csv")
data
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

data.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(data.corr(),annot = True, linewidths=0.5,fmt = '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.Speed.plot(kind = 'line',color ='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle=':')

data.Defense.plot(color = 'r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='blue')

plt.xlabel('Attack')

plt.ylabel('Defence')

plt.title('Attack Defense Scater Plot')

plt.show()
data.Speed.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
data.Speed.plot(kind='hist',bins=50)

plt.clf()