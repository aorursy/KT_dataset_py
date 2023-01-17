# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df2015 = pd.read_csv("../input/2015.csv")

df2016 = pd.read_csv("../input/2015.csv")

df2017 = pd.read_csv("../input/2015.csv")

df2015.head()
df2015["Happiness Score"].replace(['-|0'],0.0,inplace = True)

df2015 = df2015.iloc[:100,:]

area_list =list(df2015["Country"].unique())

plt.figure(figsize = (15,10))

sns.barplot(x = area_list,y = df2015["Happiness Score"])

plt.xticks(rotation = 90)

plt.xlabel('Country')

plt.ylabel('Happines Score')

plt.title('world happiness score in 2015')
df=df2015.copy()
df["Freedom"] = df2015["Freedom"]/max(df2015["Freedom"])

df["Family"] =df2015["Family"]/max(df2015["Family"])

df["Economy (GDP per Capita)"] =df2015["Economy (GDP per Capita)"]/max(df2015["Economy (GDP per Capita)"])

df["Happiness Score"] =df2015["Happiness Score"]/max(df2015["Happiness Score"])

data = pd.concat([df, df["Freedom"], df["Family"], df["Economy (GDP per Capita)"], df["Happiness Score"]])

data.sort_values(["Happiness Score","Family","Economy (GDP per Capita)","Happiness Score"],axis =0,ascending =[False,False,False,False],inplace =True)

#data.sort_values(["Happiness Score"],axis =0,ascending =[False],inplace =True)

plt.figure(figsize =(15,20))

sns.pointplot(x = data.Country[:60] ,y =data["Freedom"], color = 'lime',alpha = 0.8)

sns.pointplot(x = data.Country[:60] ,y =data["Family"], color = 'red',alpha = 0.8)

sns.pointplot(x = data.Country[:60] ,y =data["Economy (GDP per Capita)"], color = 'cyan',alpha = 0.8)

sns.pointplot(x = data.Country[:60] ,y =data["Happiness Score"], color = 'blue',alpha = 0.8)

plt.text(40,1.02,'Freedom',color ='lime')

plt.text(40,1.01,' Family',color = 'red')

plt.text(40,1.00,'Economy (GDP per Capita)',color ='cyan')

plt.text(40,0.99,'Happiness Score',color ='blue')

plt.xlabel ("Countries")

plt.xticks(rotation =90)

plt.ylabel("Normalized values")

plt.title("Comparization")

plt.show()
data.head()
sns.jointplot(data.Freedom,data["Happiness Score"],kind ="kde",size = 10)

plt.show()
data.columns
sns.jointplot(data["Happiness Score"],data['Generosity'],kind = 'kde',size = 10)

plt.show()
sns.jointplot(data["Happiness Score"],data["Health (Life Expectancy)"],kind ="reg")

plt.show()
sns.jointplot(df2015["Happiness Score"],df2015["Economy (GDP per Capita)"],kind = "reg",size=10, ratio=5, color="r")

plt.show()
sns.jointplot(data["Happiness Score"],data["Economy (GDP per Capita)"],kind = "reg",size=10, ratio=5, color="r")

plt.show()
df_fin = df2015[df2015["Country"] == 'Finland']

df_fin.drop(["Standard Error","Dystopia Residual","Region","Happiness Rank"],inplace = True,axis = 1) 
df_fin
data.head()
df2016.head()
sns.lmplot(x= "Happiness Score",y = "Family",data = data)

plt.show()
f,ax =plt.subplots(figsize = (10,10))

sns.heatmap(data.corr(),annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
data.head()
plt.figure(figsize=(15,10))

data.Region.unique()

sns.boxplot(x = "Region", y = "Happiness Score",data = data,palette="PRGn")

plt.xticks(rotation = 90)

plt.show()
sns.countplot(data.Region)

plt.title("region",color = 'blue',fontsize=15)

plt.xticks(rotation = 90)