# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/avocado.csv')

df.head()
df['region'].unique()
df['year'].unique()
df['type'].unique()
df.tail()
hou = df.loc[df['region']=='Houston',]
hou.head()
g = sns.catplot(data=hou, kind='box', x='year', y='AveragePrice', palette='winter')

g.fig.suptitle("Houston: Avocado prices")
g = sns.catplot(data=hou, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')

g.fig.suptitle("Houston: Average avocado prices")
g = sns.catplot(data=hou, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')

g.fig.suptitle("Houston: Average Avocado prices")
indy = df.loc[df['region']=='Indianapolis',]
indy.head()
g = sns.catplot(data=indy, kind='box', x='year', y='AveragePrice', palette='winter')

g.fig.suptitle("Indianapolis: Avocado prices")
g = sns.catplot(data=indy, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')

g.fig.suptitle("Indianapolis: Average avocado price")
g = sns.catplot(data=indy, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')

g.fig.suptitle("Indianapolis: Average avocado price")
chicago = df.loc[df['region']=="Chicago",]

chicago.head()
g = sns.catplot(data=chicago, kind='box', x='year', y='AveragePrice', palette='winter')

g.fig.suptitle("Chicago: Avocado prices")
g = sns.catplot(data=chicago, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')

g.fig.suptitle("Chicago: Average avocado price")
g = sns.catplot(data=chicago, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')

g.fig.suptitle("Chicago: Average avocado price")
GrandRapids = df.loc[df['region']=="GrandRapids",]

GrandRapids.head()
g = sns.catplot(data=GrandRapids, kind='box', x='year', y='AveragePrice', palette='winter')

g.fig.suptitle("Grand Rapids: Avocado prices")
g = sns.catplot(data=GrandRapids, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')

g.fig.suptitle("Grand Rapids: Average avocado price")
g = sns.catplot(data=GrandRapids, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')

g.fig.suptitle("Grand Rapids: average avocado price")
yr2017 = df.loc[df['year'].isin(['2017'])]

yr2017.head()
g = sns.catplot(data=yr2017, kind='swarm', palette='magma', x='type', y='AveragePrice', hue='region')
totalUS = df.loc[df['region'].isin(['TotalUS'])]

totalUS.head()
g = sns.catplot(data=totalUS, kind='box', x='year', y='AveragePrice', palette='winter')

g.fig.suptitle("Total US: Avocado prices")
g = sns.catplot(data=totalUS, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')

g.fig.suptitle("Total US: Average avocado price")
g = sns.catplot(data=totalUS, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')

g.fig.suptitle("Total US: average avocado price")