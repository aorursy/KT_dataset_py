# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/us-monthly-unemployment-rate-1948-present/USUnemployment.csv")

data.head()
print("Min: ",data["Jan"].min())

print("Max: ",data["Jan"].max())
import matplotlib.pyplot as plt

import seaborn as sns
sns.lineplot(x="Year", y="Jan",data=data)
for colname in data.columns:

    if colname== "Year":

        continue

    sns.lineplot(x="Year",y=colname,data=data)

    plt.show()
t4to10=data[(data["Year"]>=2004) & (data["Year"]<2011)]

for colname in data.columns:

    if colname== "Year":

        continue

    sns.lineplot(x="Year",y=colname,data=t4to10)

    plt.show()
means=[]

for year in data["Year"]:

    specific_year=data[data["Year"]==year]

    tempsum=0

    for colname in data.columns:

        if colname== "Year":

            continue

        tempsum+=float(specific_year[colname])

    means.append(tempsum/12)         
data["Means"]=means
sns.lineplot(x="Year",y="Means",data=data)

plt.show()
data.columns
summer_means=[]

for year in data["Year"]:

    specific_year=data[data["Year"]==year]

    tempsum=0

    for colname in ["Jun","Jul","Aug"]:

        if colname== "Year":

            continue

        tempsum+=float(specific_year[colname])

    summer_means.append(tempsum/3)

data["summermeans"]=summer_means    
sns.lineplot(x="Year",y="summermeans",data=data)

plt.show()
difeerence=sum(abs(data["Means"]-data["summermeans"]))

print(difeerence)

print("Year wise: ",difeerence/75)
winter_means=[]

for year in data["Year"]:

    specific_year=data[data["Year"]==year]

    tempsum=0

    for colname in ["Jan","Feb","Dec"]:

        if colname== "Year":

            continue

        tempsum+=float(specific_year[colname])

    winter_means.append(tempsum/3)

data["wintermeans"]=winter_means
sns.lineplot(x="Year",y="wintermeans",data=data)

plt.show()
difeerence=sum(abs(data["Means"]-data["wintermeans"]))

print(difeerence)

print("Year wise: ",difeerence/75)
difeerence=sum(abs(data["summermeans"]-data["wintermeans"]))

print(difeerence)

print("Year wise: ",difeerence/75)
tmodern=data[data["Year"]>=2009]

sns.lineplot(x="Year",y="Means",data=tmodern)

tmodern=data[(data["Year"]<=2009) & (data["Year"]>1999)]

sns.lineplot(x="Year",y="Means",data=tmodern)
tmodern=data[(data["Year"]<=1999) & (data["Year"]>1989)]

sns.lineplot(x="Year",y="Means",data=tmodern)
tmodern=data[(data["Year"]<=1990) & (data["Year"]>1979)]

sns.lineplot(x="Year",y="Means",data=tmodern)