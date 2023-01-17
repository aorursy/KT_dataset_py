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
data = pd.read_csv ('../input/pokemon_alopez247.csv')
data.info()

data.corr()
f,ax = plt.subplots(figsize=(13,13))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)
data.columns
data.head(10)
data.columns
data.Attack.plot(kind='line', color='g', label='Attack', linewidth=1, alpha=0.5, grid=True, linestyle=':')
data.Defense.plot(color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('def-at line plot')
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='blue')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack Defense Scatter Pilot')
data.columns
data.Speed.plot(kind='hist',bins = 50,figsize =(15,15))
