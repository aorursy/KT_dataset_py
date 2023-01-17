import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from mpl_toolkits import mplot3d
print(os.listdir("../input"))
data=pd.read_csv("../input/vgsales.csv")
data.head(6)
data.describe()
data.info()
AvgNASls= data.groupby('Genre')['NA_Sales'].mean()

AvgNASls
SumNASls= data.groupby('Genre')['NA_Sales'].sum()

SumNASls
ax=sns.countplot(x = "Genre", data=data) 

plt.xticks(rotation=45)

ax.set_title(label='Number of Genres', fontsize=15)
ax=sns.barplot(x='Genre',y='NA_Sales', data=data) 

plt.xticks(rotation=90)

ax.set_title(label='Average NA_Sales', fontsize=15)
ax=sns.barplot(x='Genre',y='Global_Sales', data=data) 

plt.xticks(rotation=90)

ax.set_title(label='Average Global Sales', fontsize=15)
ax=sns.barplot(x="Year", y="Other_Sales", data=data)

plt.xticks(rotation=90)

ax.set_title(label='Average Other Sales', fontsize=15)
sns.barplot(x='Year',y='Global_Sales', data=data) 

plt.xticks(rotation=90)
sns.barplot(x="Year", y="Other_Sales", data=data,ci=20)

plt.xticks(rotation=90)
sns.barplot(x="Year", y="Other_Sales", data=data,ci=100)

plt.xticks(rotation=90)
sns.scatterplot("Rank","NA_Sales",data=data)
sns.scatterplot(x="Year", y="EU_Sales", data=data);
sns.scatterplot(x="Year", y="EU_Sales",hue='Genre',data=data);
sns.catplot(x="Genre", y="NA_Sales",data=data)

plt.xticks(rotation=90)
sns.catplot(x="Genre", y="NA_Sales",hue='Year',data=data)

plt.xticks(rotation=90)
sns.catplot(x="Genre", y="Year", kind="box", data=data)

plt.xticks(rotation=45);
sns.catplot(x='Year',y='NA_Sales', col='Genre',col_wrap=3,data=data,kind='bar',height=4,aspect=2)
ax=sns.lineplot(x="Year", y="Global_Sales",ci=None, data=data);

ax.set_title(label='Global sales/year', fontsize=15)
sns.lineplot(x="Year", y="Global_Sales",ci='sd', data=data);
sns.lineplot(x="Year", y="Global_Sales",estimator=None, data=data);
sns.lineplot(x="Year", y="Global_Sales",hue='Genre', data=data);
sns.lineplot(x="Genre", y="NA_Sales",hue='Year', data=data)

plt.xticks(rotation=45);
ax=sns.pointplot(x="Genre", y="NA_Sales", data=data)

plt.xticks(rotation=45);

ax.set_title(label='NA Sales', fontsize=15)
sns.pointplot(x="Genre", y="NA_Sales",hue='Year', data=data)

plt.xticks(rotation=45);
sns.pointplot(x="Year", y="JP_Sales", data=data)

plt.xticks(rotation=90);
sns.pointplot(x="Year", y="JP_Sales",hue='Genre', data=data)

plt.xticks(rotation=90);
sns.pointplot(x="Year", y="JP_Sales",join=False, data=data)

plt.xticks(rotation=90);
from numpy import median

sns.pointplot(x="Year", y="JP_Sales", data=data,estimator=median)

plt.xticks(rotation=90);
sns.pointplot(x="Year", y="JP_Sales", data=data,ci='sd')

plt.xticks(rotation=90);
sns.pointplot(x="Year", y="JP_Sales", data=data,capsize=1)

plt.xticks(rotation=90);
sns.distplot(data['NA_Sales'],hist=True,bins=5)
sns.distplot(data['Global_Sales'],hist=True,bins=5)
sns.distplot(data['Global_Sales'],kde=True,rug=True,bins=5)
sns.pairplot(data)
fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

ax.scatter(data['NA_Sales'], data['JP_Sales'],data['EU_Sales'])

ax.set_xlabel('NA_Sales')

ax.set_ylabel('JP_Sales')

ax.set_zlabel('EU_Sales')

ax.set_title(label='3D Plot', fontsize=15)

plt.show()
data.corr()

sns.heatmap(data.corr())