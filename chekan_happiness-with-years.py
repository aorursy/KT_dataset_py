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
data2015=pd.read_csv("../input/2015.csv")
data2016=pd.read_csv("../input/2016.csv")
data2017=pd.read_csv("../input/2017.csv")
data2015.info()
data2016.info()
data2017.info()
data2015.shape
data2015.head(10)
data2016.shape
data2016.head(10)
data2017.shape
data2017.head(10)
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data2017.corr(),fmt='.1f',annot=True,ax=ax)
plt.title("Heatmap of Correlation 2017")
data2017.plot(kind='scatter',figsize=(10,10),x='Economy..GDP.per.Capita.',y='Health..Life.Expectancy.')
plt.title("Economy vs Healt Life Expectancy")
data2017.plot(kind='line',figsize=(10,10),x='Happiness.Rank',y='Generosity')
plt.title("Happiness Rank vs Generosity")
f, ax = plt.subplots(figsize=(15, 15))
sns.barplot(x=data2015["Happiness Score"][0:10], y=data2015["Country"][0:10],label="2015", color="b")
sns.barplot(x=data2016["Happiness Score"][0:10], y=data2016["Country"][0:10],label="2016", color="r")
sns.barplot(x=data2017["Happiness.Score"][0:10], y=data2017["Country"][0:10],label="2017", color="y")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 8), ylabel="",
       xlabel="First 10 Country happiness per year")
sns.despine(left=True, bottom=True)
f,ax=plt.subplots(figsize=(30,20))
sns.barplot(x=data2015["Region"],y=data2015["Happiness Score"],ax=ax,color="g",label="2015")
sns.barplot(x=data2016["Region"],y=data2016["Happiness Score"],ax=ax,color="y",label="2016")
ax.legend(ncol=1,loc="upper right",frameon=True)
ax.set(xlabel="Regions vs Happiness Score")