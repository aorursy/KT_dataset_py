# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn  as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patch

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
Data=pd.read_csv("../input/vgsales.csv")

# Any results you write to the current directory are saved as output.
Data.head()
#Let's see PLatform First
#Wii is most used platform
Platform=Data["Platform"]
plt.figure(figsize=(8,6))
sns.barplot(Platform.value_counts(),Platform.unique())
plt.xlabel("Platform")
plt.ylabel("Total No. of games")
plt.show()
#Let's see the total no of games produced in each year
#Year 2009 was most successful year
plt.figure(figsize=(25,15))
Year=Data["Year"]
type(Year[0])
Year=Year.value_counts()
Year=pd.DataFrame(Year)
plt.bar(Year.index,Year["Year"])
plt.xticks(Year.index)
plt.xlabel("Total no of games produced")
plt.ylabel("Year")
plt.show()
#Let's have some Genre
#So Sports games are produced in highest numbers
Genre=Data["Genre"]
plt.figure(figsize=(12,10))
sns.barplot(Genre.unique(),Genre.value_counts(),palette="deep")
plt.show()
# Here we have bar chart of companies with their total number of games produced.
#So EA games top the chart.
#Here we are showing only company names who has published more than 40 games.
Publisher=Data["Publisher"]
Publisher=Publisher.dropna()
len(Publisher.unique())
Publisher_Data=pd.DataFrame(Publisher.value_counts())
Publisher_Data.columns=["Total no of Games"]
len(Publisher_Data[Publisher_Data["Total no of Games"]>40])
Publisher_Data_51=Publisher_Data[Publisher_Data["Total no of Games"]>40]
plt.figure(figsize=(10,15))
sns.barplot(Publisher_Data_51["Total no of Games"],Publisher_Data_51.index,palette="deep")
plt.show()
#Lets find number of games who sold more than 100k copies in each year
#Total 1431 different type of games of more than 100k copies sold each.
x_data=[]
Year1=[]
for i,j in Data.groupby("Year"):
    x_data.append([j.shape[0]])
    Year1.append(i)
plt.figure(figsize=(25,10))
plt.plot(Year1,x_data,color="brown")
plt.xlabel("Year")
plt.xticks(Year.index)
plt.ylabel("Total no of Games who sold more than 100k copies")
plt.show()
x_data=[]
for i,j in Data.groupby("Year"):
    z=[]
    #print(i)
    #print(j["Genre"].value_counts())
    z=[i,j["Genre"].value_counts().index[0],j["Genre"].value_counts()[0]]
    #x_data.append({j["Genre"].value_counts().index[0]:j["Genre"].value_counts()[0]})
    x_data.append(z)
x_data1=pd.DataFrame(x_data)
x_data1.columns=["Year","Genre","Total no of games"]

#Here we have created barchart of game genre which produced maximum number of games in a particula year.
#Nowadays action movie are more famous.
x_data1["Year"]=[int(i) for i in x_data1["Year"] ]
x_data2=x_data1
x_data2=x_data2.drop(x_data2.tail(2).index)
x_data2=x_data2.reindex()
#x_data2.loc[x_data2["Genre"=="Role-Playing",:]]
fig, ax = plt.subplots(figsize=(15, 10))

ax.bar(x_data2["Year"],x_data2["Total no of games"],color="y")
i=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(),height+20,x_data2["Genre"][i],rotation=90)
    i=i+1
plt.xticks(x_data2["Year"],rotation=90)
plt.ylabel("Total no. of games")
plt.show()
#Here we have created the barchart of companies who produced most number of successful games in a particular year.
x_data=[]
for i,j in Data.groupby("Year"):
    z=[]
    z=[i,j["Publisher"].value_counts().index[0],j["Publisher"].value_counts()[0]]
    x_data.append(z)
x_data1=pd.DataFrame(x_data)
x_data1.columns=["Year","Publisher","Total no of games"]   
x_data1

x_data1["Year"]=[int(i) for i in x_data1["Year"] ]
x_data2=x_data1
x_data2=x_data2.drop(x_data2.tail(2).index)
x_data2=x_data2.reindex()
fig, ax = plt.subplots(figsize=(15, 10))
ax.bar(x_data2["Year"],x_data2["Total no of games"],color="g",alpha=0.6)
i=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(),height+10,x_data2["Publisher"][i],rotation=90)
    i=i+1
plt.xticks(x_data2["Year"],rotation=90)
plt.ylabel("Total no. of games")
plt.show()
#These DataFrames contains list of games who sold maximum copies each year

#US
Data_NA=pd.DataFrame(columns=Data.columns)
for i,j in Data.groupby("Year"):
    z=pd.DataFrame(j[j["NA_Sales"]==max(j["NA_Sales"])])
    Data_NA=pd.concat([Data_NA,z],axis=0)
Data_NA=Data_NA.loc[:,["Year","Name","NA_Sales"]]
Data_NA.columns=["Year","NA-Game Names","NA_Sales"]
Data_NA=Data_NA.set_index("Year")
Data_NA

#Europe
Data_EU=pd.DataFrame(columns=Data.columns)
for i,j in Data.groupby("Year"):
    z=pd.DataFrame(j[j["EU_Sales"]==max(j["EU_Sales"])])
    Data_EU=pd.concat([Data_EU,z],axis=0)
Data_EU=Data_EU.loc[:,["Year","Name","EU_Sales"]]
Data_EU.columns=["Year","EU-Game Names","EU_Sales"]
Data_EU=Data_EU.set_index("Year")
Data_EU

#Japan    
Data_JP=pd.DataFrame(columns=Data.columns)
for i,j in Data.groupby("Year"):
    z=pd.DataFrame(j[j["JP_Sales"]==max(j["JP_Sales"])])
    Data_JP=pd.concat([Data_JP,z],axis=0)
Data_JP=Data_JP.loc[:,["Year","Name","JP_Sales"]]
Data_JP.columns=["Year","JP-Game Names","JP_Sales"]
Data_JP=Data_JP.set_index("Year")
Data_JP

#Others
Data_Others=pd.DataFrame(columns=Data.columns)
for i,j in Data.groupby("Year"):
    z=pd.DataFrame(j[j["Other_Sales"]==max(j["Other_Sales"])])
    Data_Others=pd.concat([Data_Others,z],axis=0)
Data_Others=Data_Others.loc[:,["Year","Name","Other_Sales"]]
Data_Others.columns=["Year","Others-Game Names","Other_Sales"]
Data_Others=Data_Others.set_index("Year")
Data_Others
#World
Data_Global=pd.DataFrame(columns=Data.columns)
for i,j in Data.groupby("Year"):
    z=pd.DataFrame(j[j["Global_Sales"]==max(j["Global_Sales"])])
    Data_Global=pd.concat([Data_Global,z],axis=0)
Data_Global=Data_Global.loc[:,["Year","Name","Global_Sales"]]
Data_Global.columns=["Year","Global-Game Names","Global_Sales"]
Data_Global=Data_Global.set_index("Year")



Data_NA
Data_EU
Data_JP
Data_Others
Data_Global






