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
data_2017 = pd.read_csv('../input/2017.csv')

data_2016 = pd.read_csv('../input/2016.csv')

data_2015 = pd.read_csv('../input/2015.csv')
data_2017.columns
data_2017.info()
data_2017.corr()
f,ax = plt.subplots(figsize=(14,14))

sns.heatmap(data_2017.corr(), annot=True, linewidths=.5, fmt='.2f',ax=ax, cmap="YlGn")
data_2017.head(10)
data_2017.tail(10)
data_2015.head(10)
data_2016.head(10)
x = data_2017['Family']<0.5

data_2017[x]
df = data_2015[(data_2015['Region']=='Western Europe')]

df.nsmallest(2, 'Health (Life Expectancy)')
data_2015.Generosity.plot(kind = 'hist',bins = 150,figsize = (14,14),color='green')

plt.show()
data_2015.plot(kind='scatter', x='Happiness Rank', y='Freedom',alpha = 0.9,color = 'green')

plt.xlabel('Happiness Rank')              # label = name of label

plt.ylabel('Freedom')

plt.title('Happiness Rank over Freedom Scatter Plot')   
data_2015[(data_2015['Freedom']>0.6) & (data_2015['Happiness Rank']>140)]