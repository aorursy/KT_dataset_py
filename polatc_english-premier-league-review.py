# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visulation tool

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/epldata_final.csv")

data.info() #look data info

data.head(3) #look data head

data.tail(4) #look data tail
data.columns #View column names

f,ax = plt.subplots(figsize=(10,10)) #f is figure,ax is axis

sns.heatmap(data.corr(),annot=True,linewidth=1,fmt=".2f",ax=ax) #

plt.show() #Show plot
data.market_value.plot(kind="line",color="blue",label="market value",linewidth=3,grid=True,alpha=0.7,linestyle="-")

data.age.plot(color="red",label="age",linewidth=3,grid=True,alpha=0.7,linestyle=":")

plt.xlabel("x label") #x label

plt.ylabel("y label") #y label

plt.title("Line plot") #title

plt.legend("upper left") #label location

plt.show() #Show plot
#Scatter plot

data.plot(kind="scatter",x="market_value",y="big_club",alpha=0.5,color="red")

plt.xlabel("market value")

plt.ylabel("big club")

plt.title("Big clup market value compare")

plt.show()
#Histogram plot

data.market_value.plot(kind="hist",bins=20,figsize=(10,10))

plt.title("Market value histogram")

plt.show()
#clear figure 

plt.clf()
#Filtering with pandas

value = data["market_value"]>40

age = data["age"]<25

data[age & value]