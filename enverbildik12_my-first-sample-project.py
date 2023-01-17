# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/globalterrorismdb_0718dist.csv",engine='python')
data.head(5)
data.info()
data.describe()
data.corr()
#convert all upper case names of columns to lower case
data.columns=[each.lower() for each in data.columns]
#Bar plot
data_bar1=data.iloc[2:8,:].country_txt.unique()
data_bar2=data.iloc[2:8,:].country.unique()
plt.bar(data_bar1,data_bar2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
data.count()
(data.columns)
#data correlation
f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True,linewidth=1,linecolor="white",alpha=.5,fmt=".2f",ax=ax)
plt.show()
#Line plot
#line plat is better when x axis is time
data.region.plot(kind="line",color="r",label="region",linewidth=1,alpha=.5,grid=True,linestyle=":")
data.latitude.plot(color="b",label="latitude",linewidth=1,alpha=.5,grid=True,linestyle="-.")
plt.legend(loc="upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
#Scatter Plot
#Scatter is better when there is a correlation between two variable
data.plot(kind="scatter",x="attacktype1",y="attacktype2",alpha=.5,color="r")
plt.xlabel("attacktype1")
plt.ylabel("attacktype2")
plt.title("relationships between attacktype1 and attacktype2")

#Histogram
#Histogram is better when we need to see disribution of numerical data.
#bins=number of bar in figure
data.attacktype1.plot(kind="hist",bins=50,figsize=(12,12))
plt.show()
