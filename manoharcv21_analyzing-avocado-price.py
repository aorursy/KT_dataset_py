###### This Python 3 environment comes with many helpful analytics libraries installed
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

Df1 =  pd.read_csv("../input/avocado.csv")
avo  = Df1["Date"]
print(avo)
avo1 = Df1[["Small Bags","Large Bags"]]
type(avo1)
Df1.Date.value_counts()
Df1.region.value_counts()
Df1.head()
#Maximum aveg pric of avo
Df1["AveragePrice"].max()
Df1["AveragePrice"].min()
# what was the avg total volume of Avo pe year
Df1.groupby('year')['Total Volume'].max()
Df1.groupby('year')['AveragePrice'].max()
#Avg price by region 
Df1.groupby('region')['AveragePrice'].max()
#Avg price of the Albnay region..
Df1[Df1['region']=='Albany'][['AveragePrice','region','year','4046','4225','4770']]
#Organic Avocado's price in 2018 of type 4046
Organic_Avo = Df1[(Df1['type']=='organic') & (Df1['year']== 2018)][['AveragePrice','region','year','4046']]
Organic_Avo
O_Avo = Df1[Df1['type'] == 'organic']
O_Avo
#Organic Avg price by years 
O_Avo.groupby('year')['AveragePrice'].mean()
#Organic Avo Avg price by regions 
O_Avo.groupby('region')['AveragePrice'].mean().sort_values(ascending = False)
# Count of type's of organic by Year
Df1.groupby('year')['type'].value_counts()
#
Df1[Df1['AveragePrice']>Df1['AveragePrice'].mean()][['AveragePrice','year','region','4046','4225','4770']]
Df1[(Df1["year"] == 2015) & (Df1['type'] == 'organic')]
Df1.iloc[-1]
dfg = Df1.groupby("Date")
print(type(dfg))
print(dfg)
dfg.nunique()
dfg1=dfg.aggregate('AveragePrice')
dfg1.sum()
Df1.groupby('type').describe()
Avo_Type = 'organic'
Df1[Df1.type == Avo_Type]

regions = Df1.groupby('region')
print("Total regions :", len(regions))
print("-------------")
for name, group in regions:
    print(name, " : ", len(group))
#conver the data type of date 
Df1["Date"]= pd.to_datetime(Df1["Date"])
Df1["Date"].sample
plt.figure(figsize=(20,9))
plt.title("Distrubtion of Average Price")
Av = sns.distplot(Df1["AveragePrice"])
plt.figure(figsize=(16,9))
plt.title("Boxplot")
Av = sns.boxplot(Df1["AveragePrice"])
plt.figure(figsize=(10,9))
plt.title("Avg.Price of Avocado by Type")
Av= sns.barplot(x="type",y="AveragePrice",data= Df1,palette = 'pink')
plt.figure(figsize=(10,11))
plt.title("Avg.Price of Avocado by Region")
Av= sns.barplot(x="AveragePrice",y="region",data= Df1)
Organic_Avo = Df1[(Df1['type']=='organic')]
Organic_Avo = Organic_Avo.sort_values("AveragePrice")
Organic_Avo
plt.title(" Avg.Price of organice Avocado ")
org= sns.factorplot("AveragePrice","region",data = Organic_Avo,hue='year', # category
                   height=13,
                   aspect=0.8,
                   palette='muted',
                   join=False)

plt.figure(figsize=(10,9))
plt.title("Date / Average Price ")
av = sns.tsplot(data=Df1, time="Date", unit="region",condition="type", value="AveragePrice") 