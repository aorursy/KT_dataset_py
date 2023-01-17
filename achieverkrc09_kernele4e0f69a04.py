import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from mpl_toolkits import mplot3d
os.chdir("../input")
dt=pd.read_csv("vgsales.csv")
dt.head(6)
dt.describe()
dt.describe()
dt.dtypes
dt.tail()
dt.shape
#Number of Genres

sns.countplot(x = "Genre", data=dt)  

plt.xticks(rotation=90)
#barplot showing average of NA_Sales for each Genre

sns.barplot(x='Genre',y='NA_Sales', data=dt) 

plt.xticks(rotation=90)
#barplot showing average of Global_Sales for each Genre

sns.barplot(x='Genre',y='Global_Sales', data=dt) 

plt.xticks(rotation=90)
#Scatter plot to show how NA_Sales disrtibuted on each Rank

sns.scatterplot("Rank","NA_Sales",data=dt)
#Categorial plot to show NA_Sales for each Genre

sns.catplot(x="Genre", y="NA_Sales",data=dt)

plt.xticks(rotation=90)
#Categorial plot to show NA_Sales for each Genre subdevided for each year

sns.catplot(x="Genre", y="NA_Sales",hue='Year',data=dt)

plt.xticks(rotation=90)
#barplot showing average of Global_Sales for each year

sns.barplot(x='Year',y='Global_Sales', data=dt) 

plt.xticks(rotation=90)
#Using Groupby, Group average NA_Sales by Genre

Avg_NA_Sls= dt.groupby('Genre')['NA_Sales'].mean()

Avg_NA_Sls
#Using Groupby, Group sum of NA_Sales by Genre

Sum_NA_Sls= dt.groupby('Genre')['NA_Sales'].sum()

Sum_NA_Sls
#plot sales for each Year with Genre using boxplot 

sns.catplot(x="Genre", y="Year", kind="box", data=dt)

plt.xticks(rotation=90);
#Scatter plot showing the EU_Sales for each year

sns.scatterplot(x="Year", y="EU_Sales", data=dt);
#Scatter plot showing the EU_Sales for each year, subdeviding by Genre 

sns.scatterplot(x="Year", y="EU_Sales",hue='Genre',data=dt);
# Global_Sales for each year

sns.lineplot(x="Year", y="Global_Sales",ci=None, data=dt);
#Line plot showing the Global_Sales for each year

sns.lineplot(x="Year", y="Global_Sales",ci='sd', data=dt);

#Line plot showing the Global_Sales for each year.Using Estimator

sns.lineplot(x="Year", y="Global_Sales",estimator=None, data=dt);
#Line plot showing the Global_Sales for each year.Using Genre as hue parameter

sns.lineplot(x="Year", y="Global_Sales",hue='Genre', data=dt);
#Line plot showing the NA_Sales for each Genre.Using Year as hue parameter

sns.lineplot(x="Genre", y="NA_Sales",hue='Year', data=dt)

plt.xticks(rotation=90);
#Point plot showing the NA_Sales for each Genre.

sns.pointplot(x="Genre", y="NA_Sales", data=dt)

plt.xticks(rotation=90)
#Point plot showing the NA_Sales for each Genre. Using Time as hue parameter.

sns.pointplot(x="Genre", y="NA_Sales",hue='Year', data=dt)

plt.xticks(rotation=90);
#Point plot showing the JP_Sales for each Year. 

sns.pointplot(x="Year", y="JP_Sales", data=dt)

plt.xticks(rotation=90);
#Point plot showing the JP_Sales for each Year. Using Genre as hue parameter.

sns.pointplot(x="Year", y="JP_Sales",hue='Genre', data=dt)

plt.xticks(rotation=90);
#Point plot showing the JP_Sales for each Year. Without joining

sns.pointplot(x="Year", y="JP_Sales",join=False, data=dt)

plt.xticks(rotation=90);
#Point plot showing the JP_Sales for each Year. With estimator as median

from numpy import median

sns.pointplot(x="Year", y="JP_Sales", data=dt,estimator=median)

plt.xticks(rotation=90);
#Point plot showing the JP_Sales for each Year. With ci as standard deviation

sns.pointplot(x="Year", y="JP_Sales", data=dt,ci='sd')

plt.xticks(rotation=90);
#Point plot showing the JP_Sales for each Year. Using the capsize

sns.pointplot(x="Year", y="JP_Sales", data=dt,capsize=1)

plt.xticks(rotation=90);
#Bar plot to show Other sales for each year

sns.barplot(x="Year", y="Other_Sales", data=dt)

plt.xticks(rotation=90)
#Bar plot to show Other sales for each year, with ci as 100

sns.barplot(x="Year", y="Other_Sales", data=dt,ci=100)

plt.xticks(rotation=90)
#Bar plot to show Other sales for each year, with si as 20

sns.barplot(x="Year", y="Other_Sales", data=dt,ci=20)

plt.xticks(rotation=90)
#histogram showing the NA_Sales

sns.distplot(dt['NA_Sales'],hist=True,bins=5)
#histogram showing the Global_Sales

sns.distplot(dt['Global_Sales'],hist=True,bins=5)
#histogram showing the Global_Sales using kde and rug

sns.distplot(dt['Global_Sales'],kde=True,rug=True,bins=5)
#catplot to show NA_Sales for each year for each Genre

sns.catplot(x='Year',y='NA_Sales', col='Genre',col_wrap=4,data=dt,kind='bar',height=4,aspect=.8)
#3D plot to show NA Sales, JP Sales and EU Sales

fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

ax.scatter(dt['NA_Sales'], dt['JP_Sales'],dt['EU_Sales'])

ax.set_xlabel('NA_Sales')

ax.set_ylabel('JP_Sales')

ax.set_zlabel('EU_Sales')

plt.show()