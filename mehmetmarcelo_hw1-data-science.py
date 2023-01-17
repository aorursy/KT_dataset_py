# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data
data.columns
len(data.columns)
data.info()
print(data.describe())
#ilk 20 ülkenin datası şu şekilde:

data.head(20)
data[data["Country or region"].str.contains("Turkey")]
data1=data.drop(["Overall rank"],axis=1)

data1.plot(figsize=(13,13))

plt.show()
#Correlation map

data.corr()
f,ax=plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(),annot=True,lw=.5,fmt=".1f",ax=ax)

plt.show()
#scorelara göre bir line plot çizdirelim:

data.Score.plot(kind="line",color="b",label="Score",ls="--",lw=1,alpha=.7,grid=True)

plt.legend(loc="upper left")

plt.xlabel("Countries",color="r")

plt.ylabel("Score",color="r")

plt.title("Score-Line Plot",color="r")

plt.show()
data.plot(kind="scatter",x="GDP per capita",y="Social support",color="orange",alpha=.8,grid=True,figsize=(11,11))

plt.xlabel("GDP per capita",color="r")

plt.ylabel("Social support",color="r")

plt.title("GDP vs Social support",color="r")

plt.show()
#grafiğe bakınca düz bir çizgi gibi,yani aralarında doğru orantı varmış.
data.plot(kind="scatter",x="Generosity",y="GDP per capita",label="GDP vs Generosity",color="m",grid=True,figsize=(11,11))

plt.legend(loc="center right") 

plt.xlabel("Generosity",color="r")

plt.ylabel("GDP per capit",color="r")

plt.title("GDP vs Generosity",color="r")

plt.show()
f,ax=plt.subplots(figsize=(11,11))

plt.scatter(data["Freedom to make life choices"],data["Healthy life expectancy"],label="Freedom vs. Healthy",color="c",alpha=.6)

plt.legend(loc="center left")

plt.xlabel("Freedom to make life choices",color="r")

plt.ylabel("Healthy life expectancy",color="r")

plt.title("Freedom vs. Healthy",color="r")

plt.grid(True)

plt.show()
data1.plot(grid=True,alpha=.9,subplots=True,figsize=(15,15))

plt.show()
plt.subplot(2,1,1)

plt.plot(data["Overall rank"],data["Freedom to make life choices"],color="r",label="Freedom")

plt.ylabel("Freedom to make life choices",color="r")

plt.subplot(2,1,2)

plt.plot(data["Overall rank"],data["Healthy life expectancy"],color="b",label="Healthy")

plt.ylabel("Healthy life expectancy",color="b")

plt.show()
#2018 yılının verileri ise şu şekilde:

data2=pd.read_csv("/kaggle/input/world-happiness/2018.csv")

data2
#2 yıldaki mutluluk scorelarını karşılaştıralım

data.Score.plot(kind="line",color="r",label="2019",lw=1.5,alpha=.8,grid=True,figsize=(13,13))

data2.Score.plot(color="g",label="2018",lw=1.5,alpha=.8,grid=True)

plt.legend(loc="upper right")

plt.xlabel("Countries",color="y")

plt.ylabel("Score",color="y")

plt.title("Score-Line Plot",color="y")

plt.show()
#şimdi de türkiye nin iki yılını karşılaştıralım:

data[data["Country or region"].str.contains("Turkey")]

#2019 yılı
data2[data2["Country or region"].str.contains("Turkey")]

#2018 yılı
data.head()

#2019 yılı
data2.head()

#2018 yılı