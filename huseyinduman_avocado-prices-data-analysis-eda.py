# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visialization
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/avocado.csv") #read file
df.info() #quick overview of the data
df.columns = df.columns.str.replace(' ', '_')
df = df.drop('Unnamed:_0',1)
df.head()
df.Date = pd.to_datetime(df.Date)
df.head()
df.info()
sns.heatmap(df.corr(),vmax = 1, vmin = 0,annot = True)
plt.show()
df.describe()
print(df.type.nunique()) #number of different types
print(df.year.nunique()) #number of different year
print(df.region.nunique()) #number of different regions
df.type.value_counts().plot(kind = 'bar')
plt.show()
df.type.value_counts()
df.region.value_counts()
index = np.arange(2)
objects = ['organic','conventional']
plt.bar(index,[df[df.type == "organic"].AveragePrice.mean(), df[df.type == "conventional"].AveragePrice.mean()])
plt.xticks(index,objects)
plt.show()

objects = ['organic','conventional']
plt.boxplot([df[df.type == "organic"].AveragePrice,df[df.type == "conventional"].AveragePrice])
plt.xticks([1,2],objects)
plt.show()
datayear = []
for i in df.year.unique():
    datayear.append(df[df.year == i].AveragePrice)
plt.boxplot(datayear)
plt.xticks(range(1,df.year.nunique()+1),df.year.unique())
plt.show()
datayearorganic = []
datayearconventional = []
for i in df.year.unique():
    datayearorganic.append(df[(df.year == i) & (df['type'] == 'organic')].AveragePrice.mean())
    datayearconventional.append(df[(df.year == i) & (df.type == 'conventional')].AveragePrice.mean())
bar_width = 0.35
plt.bar(np.arange(df.year.nunique()),datayearorganic,bar_width, label = 'organic')
plt.bar(np.arange(df.year.nunique())+bar_width,datayearconventional,bar_width, label = 'conventional')
plt.xticks(np.arange(df.year.nunique())+bar_width/2,df.year.unique())
plt.legend()
plt.show()
dataAvgPrice = []
dataDate =[]
dataTotalVolume=[]
for i in df.Date.dt.year.unique():
    for j in reversed(df.Date.dt.month.unique()):
        dataAvgPrice.append(df[(df.Date.dt.year == i) & (df.Date.dt.month == j)].AveragePrice.mean())
        dataDate.append(i.astype('str')+' '+j.astype('str'))
        dataTotalVolume.append(df[(df.Date.dt.year == i) & (df.Date.dt.month == j)].Total_Volume.mean()/(10**6))
plt.subplots(figsize = (15,15))        
plt.plot(dataDate,dataAvgPrice,label = 'price')
plt.plot(dataDate,dataTotalVolume, label = 'volume/1e6')
plt.xticks(rotation = 'vertical')
plt.legend(loc = 'best')
plt.show()

