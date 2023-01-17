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
data = pd.read_csv('../input/2017.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), annot=True, linewidths = .5, fmt='.1f', ax=ax)

plt.show()
data.head()
data.head(10)
data.columns
#draw line plot

data["Generosity"].plot(kind='line', color='blue',label='Generosity',linewidth=1,alpha=0.6,grid=True,linestyle="-")

data["Freedom"].plot(kind="line", color="green", label="Freedom", linewidth=1, alpha=1,grid=True,linestyle=":")

plt.xlabel('Generosity')

plt.ylabel('Freedom')

plt.legend(loc='upper right')

plt.show()
data.plot(kind='scatter',x='Freedom', y='Family',color='red', alpha=1,grid=True)

plt.xlabel('Freedom')

plt.ylabel('Family')

plt.title('Family Freedom Scatter Plot')

plt.show()
data["Economy..GDP.per.Capita."].plot(kind='hist',bins=50,figsize=(10,10))

plt.show()
data[(data['Economy..GDP.per.Capita.']> 1.500) & (data['Happiness.Score']> 7.0)]
print(data['Country'].index)


    