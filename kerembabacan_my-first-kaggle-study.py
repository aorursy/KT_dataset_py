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
data = pd.read_csv("../input/pokemon.csv")
data.head()
data.info()

f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=6,fmt='.1f',ax=ax)
data.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid = True,linestyle=":")
data.Defense.plot(color='r',label="Defense",linewidth=1,alpha=0.5,linestyle='-.')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Speed and Defense')

data.plot(kind='scatter', x='Attack',y='Defense',alpha=0.5,color='darkblue')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack  Defense scatter')
data.Speed.plot(kind='hist',bins=30,figsize=(15,15))
data[(data['Defense']>150) & (data['Attack']>100)]  

data.describe()
print(data["Type 1"].value_counts(dropna=False))  #Water count=112  Normal count=98
data.boxplot(column="Attack",by="Legendary")
plt.show()
data2=data
data2=pd.melt(frame=data,id_vars='Name',value_vars=['Attack','Defense'])
data2
data1 = data.head()
data2 = data.tail()
concatenateData=pd.concat([data1,data2],axis=0,ignore_index=True)
concatenateData
country=["Spain","Turkey"]
population=["11","15"]
list_label=["country","population"]
list_col=[country,population]
zipped=list(zip(list_label,list_col))
data_dict=dict(zipped)
df=pd.DataFrame(data_dict)
df
df["capital"]=["Madrid","Ankara"]
df
df["income"]=["10","11"]
df
data1=data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
plt.show()
data1.plot(subplots=True)
plt.show()
data2=data.head()
data2
import warnings
warnings.filterwarnings("ignore")

dateList=["1992-10-12","1993-11-12","1991-01-12","1992-10-12","1994-10-12"]
dateTimeObje=pd.to_datetime(dateList)
data2["Date"]=dateTimeObje
data2=data2.set_index("Date")
data2

data2.resample("B").mean()
data2.resample("A").mean()
data2
data2.resample("M").first().interpolate("linear")
data2
def div(n):
    return n/2
data.HP.apply(div)
data.HP.apply(lambda n : n*4)
data["totalPower"]=data.Attack+data.Defense
data
dataNew=data.set_index(["Type 1","Type 2"])
dataNew