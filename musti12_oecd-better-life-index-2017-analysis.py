# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import pandas as pd 

import seaborn as sns





import os

print(os.listdir("../input"))



data=pd.read_csv("../input/OECDBLI2017cleanedcsv.csv")

data.info()



data.columns

data.dtypes
data.head()

filtreData=data.iloc[:,[0,7,8,9,12,13,14,15,18,19,20,23]]

filtreData.info()
filtreData.head()
des=filtreData.describe()

print(des)
#sample boxplotting

filtreData.boxplot(column="Air pollution in ugm3",by="Water quality as pct",figsize=(20,10),fontsize=10)
plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,7],label="Water quality by country",width=0.15)

plt.rcParams["figure.figsize"] = (40,10)

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("water quality",fontsize=15,fontweight="bold")

plt.legend(fontsize=20)

plt.show()

plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,3],color="g",label="personal earnings in usd",width=0.25)

plt.rcParams["figure.figsize"] = (40,10)

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("Values of USD",fontsize=15,fontweight="bold")

plt.legend(fontsize=20)

plt.show()
plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,1],label="Employment",width=0.25)

plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,2],label="Long Term unemployment",width=0.25)

plt.rcParams["figure.figsize"] = (40,10)

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("Values of employment and unemployment",fontsize=15,fontweight="bold")

plt.legend(fontsize=18)

plt.show()
plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,4],label="Student skills as avg",width=0.25)

plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,5],label="Years in education",width=0.25,color="r")

plt.rcParams["figure.figsize"] = (40,10)

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("Values of student skills and years in education",fontsize=15,fontweight="bold")

plt.legend(fontsize=20)

plt.show()
plt.plot(filtreData.iloc[:,0],filtreData.iloc[:,6],label="Air pollution in ugm3",color="#000066")

plt.plot(filtreData.iloc[:,0],filtreData.iloc[:,7],label="Water quality as pcm",color="#660066")       

plt.rcParams["figure.figsize"] = (40,10)

plt.grid(which="major",axis="both")

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("Values of air pollution in ugm3 and water quality as pcm",fontsize=15,fontweight="bold")

plt.legend(fontsize=20)

plt.show()

plt.plot(filtreData.iloc[:,0],filtreData.iloc[:,8],label="life expentancy")

plt.rcParams["figure.figsize"] = (40,10)

plt.grid(which="major",axis="both")

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("Life expentancy in yrs",fontsize=15,fontweight="bold")

plt.legend(fontsize=20)

plt.show()
plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,8],label="life expentancy",width=0.25,color="#3333cc")

plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,10],label="life satisfaction",width=0.25,color="#00ffff")

plt.rcParams["figure.figsize"] = (40,10)

plt.xlabel("Countries",fontsize=15,fontweight="bold")

plt.ylabel("Values of life expentancy and life satisfaction",fontsize=15,fontweight="bold")

plt.legend(fontsize=20)

plt.show()
fig,ax=plt.subplots(figsize=(20,10))

sns.heatmap(filtreData.corr(),annot=True,linewidths=.10,fmt='.1f',ax=ax)

plt.plot()
plt.bar(filtreData.iloc[:,0],filtreData.iloc[:,11],label="working hours",width=0.25,color="#009933")

plt.rcParams["figure.figsize"]=(40,10)

plt.xlabel("Countries",fontweight="bold",fontsize=20)

plt.ylabel("working hours by country",fontweight="bold",fontsize=20)

plt.legend(fontsize=20)

plt.show()