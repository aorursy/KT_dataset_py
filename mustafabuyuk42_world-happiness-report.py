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
data=pd.read_csv("../input/world-happiness-report-2019.csv")
data.info()
data.head()
data.columns
data["Positive affect"].unique()
data["Positive affect"].replace(['nan'],0.0,inplace=True)

data.info()
country=list(data["Country (region)"].unique())

plt.figure(figsize=(15,10))

sns.barplot(x=country[0:70],y=freedom[0:70])

plt.xticks(rotation=90)

plt.xlabel('country')

plt.ylabel('freedom')

plt.title('freedom of countries')
data.columns
data["Freedom"]=(data["Freedom"].sort_values(ascending=False)).index.values

data['Social support']=(data['Social support'].sort_values(ascending=False)).index.values

data['Log of GDP\nper capita']=(data['Log of GDP\nper capita'].sort_values(ascending=False)).index.values
data['Freedom']=data['Freedom']/max(data['Freedom'])

data['Social support']=data['Social support']/max(data['Social support'])

data['Log of GDP\nper capita']= data['Log of GDP\nper capita']/max(data['Log of GDP\nper capita'])



plt.subplots(figsize=(15,10))

sns.pointplot(x=country[:50], y=data['Freedom'][:50], color='lime',alpha=0.8)

sns.pointplot(x=country[:50], y=data['Social support'][:50], color='red',alpha=0.7)

sns.pointplot(x=country[:50], y=data['Log of GDP\nper capita'][:50], color='green',alpha=0.6)

plt.text(30,1,'Freedom',color='lime',fontsize=17,style='italic')

plt.text(30,0.9,'Social support',color='red',fontsize=17,style='italic')

plt.text(20,0.9,'Log of GDP\nper capita',color='green',fontsize=17,style='italic')

plt.xlabel('country')

plt.xticks(rotation=90)

plt.ylabel('ratess')

plt.show()
g=sns.jointplot("Log of GDP\nper capita","Social support",data=data,kind='kde',size=5,ratio=2,color='r')
sns.jointplot("Freedom","Social support",data=data,kind='kde',size=5,ratio=2,color='r')
data.head()
data=pd.read_csv("../input/world-happiness-report-2019.csv")
data.Corruption.dropna(inplace=True)

data.Corruption.value_counts().index

sns.lmplot(x="Freedom",y="Social support",data=data)

plt.show()
data.head()
data.Corruption.dropna(inplace=True)

data.Corruption.value_counts().index
data.columns
data.Corruption.dropna(inplace=True)

data.Freedom.dropna(inplace=True)

sns.kdeplot(data.Freedom[:70],data.Corruption[:70],shade=True,cut=3)

plt.show()
f,ax=plt.subplots(figsize=(15,10))

sns.heatmap(data.corr(), annot=True,linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)