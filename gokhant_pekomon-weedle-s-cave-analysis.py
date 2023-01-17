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
data=pd.read_csv('../input/pokemon.csv')
data.info()
data.corr()
f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
data.columns
data.Speed.plot(kind='line', color = 'g', label='Speed', linewidth=1, alpha=0.5, grid=True, linestyle=':')

data.Defense.plot(color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')

plt.xlabel('Attack')

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')
data.Speed.plot(kind='hist', bins=50, figsize=(12,12))
data=pd.read_csv('../input/pokemon.csv')
series=data['Defense']

data_frame=data[['Defense']]

print(type(series))

print(type(data_frame))
x = data.Defense > 200

data[x]
data[(data.Defense > 200) & (data.Attack > 100)]
for index, value in data[['Attack']][0:5].iterrows():

    print(index, " : ", value)
treshold = sum(data.Speed) / len(data.Speed)

data['Speed_Level'] = ["High" if i > treshold else "Low" for i in data.Speed]

data.loc[:10, ["Speed_Level","Speed"]]
data=pd.read_csv('../input/pokemon.csv')

data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Type 1'].value_counts(dropna=False))
data.describe()
data.boxplot(column='Attack', by='Legendary')
data_new = data.head()

data_new
melted = pd.melt(frame=data_new, id_vars='Name', value_vars=['Attack', 'Defense'])

melted
melted.pivot(index = 'Name', columns='variable', values='value')
data1= data.head()

data2 = data.tail()

conc_data_col = pd.concat([data1, data2], axis=0)

conc_data_col
data.dtypes
data['Type 1'] = data['Type 1'].astype('category')
data.dtypes
data.info()
data['Type 2'].value_counts(dropna = False)
data1 = data

data1['Type 2'].dropna(inplace=True)
data1 = data.loc[:, ['Attack', 'Defense','Speed']]

data1.plot()
data1.plot(subplots=True)
data.describe()
import warnings

warnings.filterwarnings("ignore")

data2 = data.head()

datae_List = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

dataetime_object = pd.to_datetime(datae_List)

data2['date'] = dataetime_object

data2 = data2.set_index("date")

data2
print(data2.loc["1993-03-16"])
data = data.set_index("#")

data.head()
data.HP.apply(lambda n : n/2)
data['Total_Power'] = data.Attack + data.Defense

data.head()