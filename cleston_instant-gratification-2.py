# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

df = pd.read_csv('../input/instant-gratification/train.csv')
df0 = df[df['target'] == 0].drop(['id', 'target'], axis=1).reset_index(drop=True)

df1 = df[df['target'] == 1].drop(['id', 'target'], axis=1).reset_index(drop=True)
temp = df.drop(['id'], axis=1).reset_index(drop=True)

dfcor = temp.corr(method ='pearson')

dfcor
corlist = dfcor.apply( lambda x: (x.index[abs(x) > 0.008]).tolist() , axis=0)
principais = corlist['target']
principais.remove('target')
#plt.figure(figsize=(20,15))

#sns.scatterplot(data=df0[principais], markers = False)
temp =  df0[principais]
temp.head()
#plt.figure(figsize=(5,10))

#g = sns.PairGrid(df)

#for col in principais:

#    g.map(sns.scatterplot(data=df, y=col, x='target', hue = 'target'))
i=0

plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(5,10))

sns.scatterplot(data=df, y=principais[i], x='target', hue = 'target')

sns.boxplot(data=df, y=principais[i], x='target', hue = 'target')

i=i+1
plt.figure(figsize=(15,5))

plt.plot(df0.iloc[0][principais], color = 'red')

plt.plot(df1.iloc[0][principais], color = 'blue')
i= 100

plt.figure(figsize=(15,5))

plt.plot(df0.iloc[i][principais], color = 'red')

plt.plot(df1.iloc[i][principais], color = 'blue')
i= 1000

plt.figure(figsize=(15,5))

plt.plot(df0.iloc[i][principais], color = 'red')

plt.plot(df1.iloc[i][principais], color = 'blue')


plt.figure(figsize=(15,5))

plt.plot(df0[principais].mean(), color = 'red')

plt.plot(df1[principais].mean(), color = 'blue')