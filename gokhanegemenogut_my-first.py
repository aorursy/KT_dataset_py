# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns 
import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Iris.csv")
data.info()
del data["Id"]
data.corr()
data.columns
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.tail()
data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'SepalLength',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.SepalWidthCm.plot(color = 'r',label = 'SepalWidth',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = petallength, y = petalwidth
data.plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm',alpha = 0.5,color = 'red')
plt.xlabel('PetalLengthCm')              # label = name of label
plt.ylabel('PetalWidthCm')
plt.title('Petal Length and Width Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data['Species'].value_counts().plot(kind='bar')
plt.show()
# Histogram
# bins = number of bar in figure
data.PetalWidthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
x = data['PetalWidthCm']>1  
y= data["PetalLengthCm"]>5
data[x]
#data[x&y] #same
data[(data['PetalWidthCm']>1) & (data['PetalLengthCm']>5)]
for index,value in data[['SepalLengthCm']][0:1].iterrows():
    print(index," : ",value)
#ortalama=data.PetalLengthCm.mean()
#print(ortalama)
##type(ortalama)
ortalama=sum(data.PetalLengthCm)/len(data.PetalLengthCm)
data["buyukluk"]=["buyuk" if i>ortalama else "kucuk" for i in data.PetalLengthCm]
data.loc[70:120,["buyukluk","PetalLengthCm"]]
data= pd.read_csv("../input/Iris.csv")
data.head()

data.info()
data.SepalLengthCm[1]

data.describe()
data.boxplot(column="SepalWidthCm",by="Species")
data.SepalWidthCm[1]="NaN" #made a null cell
data.info() #now we can use dropna
data.head()
data["SepalWidthCm"].dropna(inplace = True)    #not working
data.head()       
data.dropna(subset=["SepalWidthCm"],inplace=True) #it is working
data.head()
data.shape

liste=[i for i in range(150)]
liste.remove(0)
print(liste)

data["index"]=liste
data.head()
del data["Id"]

data.head()
data=data.set_index("index") #setting index
data.head()
data=pd.read_csv("../input/Iris.csv")
data1=data.copy()
data1.head()
data1 = data.loc[:,["SepalLengthCm","SepalWidthCm","PetalWidthCm"]]
data1.plot()
data1.plot(subplots = True) #better
plt.show()
data.head()
data1.plot(kind = "scatter",x="SepalWidthCm",y = "PetalWidthCm")
plt.show()
#fig, axes = plt.subplots(nrows=2,ncols=1)
#data1.plot(kind = "hist",y = "PetalWidthCm",bins = 50,range= (0,250),normed = True,ax = axes[0])
#data1.plot(kind = "hist",y = "PetalWidthCm",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

data2=data1.head()
data2
time_list=["2012-10-20","2012-10-25","2015-08-20","2015-08-25","2015-11-26"]
type(time_list)
date_object=pd.to_datetime(time_list)
data2["date"]=date_object
data2
type(date_object)
data2=data2.set_index("date")
data2
print(data2.loc["2012-10-25"])
print(data2.loc["2012-10-25":"2015-08-25"])

data2.resample("A").mean()
#data2.resample("M").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/Iris.csv')
data= data.set_index("Id")
data.head()
data.SepalLengthCm[1]
data["SepalLengthCm"][1]
data.loc[1,["SepalLengthCm"]]
data[["SepalLengthCm","PetalLengthCm"]]
data.loc[1:10,"SepalLengthCm":"PetalLengthCm"]
data.loc[10:1:-1,"SepalLengthCm":"PetalLengthCm"]
data.loc[1:10,"SepalWidthCm":] #to the end
data.head()
filtre1=data.SepalWidthCm>3.5
filtre2=data.PetalLengthCm>3
data[filtre1&filtre2]
#column based
data.SepalLengthCm[data.PetalWidthCm>2]
def bol(n):
    return n/2

data.SepalLengthCm.apply(bol)
data.PetalLengthCm.apply(lambda n: n/2)
data.index.name = "sira"
data.head()
data.index=range(50,200,1)
data.head()
dic={"tedavi":["A","A","B","B"],"cinsiyet":["K","E","K","E"],"cevap":[40,15,50,30],"yas":[20,15,35,60]}
df=pd.DataFrame(dic)
df
df.pivot(index="tedavi",columns="cinsiyet",values="cevap")
df1=df.set_index(["tedavi","cinsiyet"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df
pd.melt(df,id_vars="tedavi",value_vars=["yas","cevap"])
df
df.groupby("tedavi").mean()
df.groupby("tedavi").yas.max() 
df.groupby("tedavi")[["yas","cevap"]].min() 
df.info()