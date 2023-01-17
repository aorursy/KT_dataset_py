%reset -f

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os
#os.chdir("F:\\Practice\\Machine Learning and Deep Learning\\Classes\\Assignment\\Kaggle\\1st")
#df=pd.read_csv("videogamesales.zip")

df=pd.read_csv('../input/vgsales.csv')
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer as ct
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
pd.options.display.max_columns = 200
df.shape
df.columns
df.dtypes
df.dtypes.value_counts()
df.head()
df.tail()
df.isnull().sum()
fig = plt.figure(figsize = (10,10))

ax1 = fig.add_subplot(121)

ax1.scatter(df["Genre"], df["Global_Sales"])

ax1.set_xlabel('Genre')

ax1.set_ylabel('Global_sales')

plt.xticks(rotation=90)
sns.catplot(x='Genre',y='Global_Sales', data=df)

plt.xticks(rotation=50)
fig = plt.figure(figsize = (10,10))

ax1 = fig.add_subplot(121)

ax1.bar(df["Genre"], df["Global_Sales"])

ax1.set_xlabel('Genre')

ax1.set_ylabel('Global_sales')

plt.xticks(rotation=90)
table = df.pivot_table('Global_Sales', index='Platform', columns='Year', aggfunc='sum')

platforms = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([platforms, sales], axis=1)

data.columns = ['Platform', 'Global Sales']



plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Platform', data=data, size=15)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Total Platform Revenue in $ Millions Per Year', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();


y = df.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.barplot(y = y, x = x)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_xticklabels(labels = x, fontsize=12, rotation=50)

ax.set_ylabel(ylabel='$ Millions', fontsize=16)

ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

plt.show();
table = df.pivot_table('Global_Sales', columns='Year', index='Name')

q = table.quantile(0.90)

data = table[table < q]

years = table.columns.astype(int)



plt.figure(figsize=(12,8))

ax = sns.boxplot(data=data)

ax.set_xticklabels(labels=years, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Revenue Per Game in $ Millions', fontsize=16)

ax.set_title(label='Distribution of Revenue Per Game by Year in $ Millions', fontsize=20)

plt.show()
sns.catplot(x="NA_Sales", y="Genre", kind="violin", bw=.05, cut=0, data=df);
table_sales = pd.pivot_table(df,values=['Global_Sales'],index=['Year'],columns=['Genre'],aggfunc='max',margins=False)



plt.figure(figsize=(20,15))

sns.heatmap(table_sales['Global_Sales'],linewidths=.5,annot=True,vmin=0.01,cmap='PuBu')

plt.title('Max Global_Sales of games')