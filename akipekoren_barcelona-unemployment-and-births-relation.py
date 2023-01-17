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
population=pd.read_csv("../input/population.csv")
accidents=pd.read_csv("../input/accidents_2017.csv")
unemployment=pd.read_csv("../input/unemployment.csv")
births=pd.read_csv("../input/births.csv")
immigrants=pd.read_csv("../input/immigrants_by_nationality.csv")
#unemployment["District Name"].value_counts()
unemployment.head()
# Firstly we will look the relation between unemployment and births.
#So we should find number of average umeployment people in districts

district_list=list(unemployment["District Name"].unique())
district_list
unemployment.Number=unemployment.Number.astype(float)
unemployment_ratio=[]
for i in district_list:
    x=unemployment[unemployment["District Name"]==i]
    ratio=sum(x.Number)/len(x)
    unemployment_ratio.append(ratio)

data=pd.DataFrame({"district_list" : district_list , "unemployment_ratio" : unemployment_ratio})
new_index=(data["unemployment_ratio"].sort_values(ascending=False)).index.values
sortedData=data.reindex(new_index)
sortedData


#visualization

plt.figure(figsize=(15,15))
sns.barplot(x=sortedData["district_list"] , y =sortedData["unemployment_ratio"])
plt.xlabel("District Names")
plt.ylabel("Number of Unemployment ")
plt.title("Average Unemployment people by Districts")    
# Secondly, we should find number of average births
births.Number=births.Number.astype(float)
births_ratio=[]
for j in district_list:
    y=births[births["District Name"]==j]
    ratio2=sum(y.Number)/len(y)
    births_ratio.append(ratio2)

data2=pd.DataFrame({"district_list" : district_list , "births_ratio" : births_ratio })
new_index2=(data2["births_ratio"].sort_values(ascending=True)).index.values
sortedData2=data2.reindex(new_index2)


# visualisation

plt.figure(figsize=(15,15))
sns.barplot(x=sortedData2["district_list"], y=sortedData2["births_ratio"])
plt.xlabel("district Names")
plt.ylabel("average of births by district ")
plt.title("average births")
sortedData["unemployment_ratio"]=sortedData["unemployment_ratio"]/max(sortedData["unemployment_ratio"])
sortedData2["births_ratio"]=sortedData2["births_ratio"]/max(sortedData2["births_ratio"])
data=pd.concat([sortedData,sortedData2["births_ratio"]],axis=1)
data.sort_values("unemployment_ratio", inplace=True)

#visualization

plt.figure(figsize=(20,10))
sns.pointplot(x="district_list" , y ="unemployment_ratio", data=data , color="blue" , alpha=0.2)
sns.pointplot(x="district_list", y="births_ratio" , data=data, color="green" , alpha=0.5)
#plt.text(40,0.6 , "average unemployment people" , color="blue")
#plt.text(40, 0.55 , "average births" , color="green")
plt.xlabel("District Names")
plt.ylabel("Numerical Values")
plt.title("Correlation between unemployment and births")
plt.grid()

#We can show correlation detailly with joint plot.
a=sns.jointplot(data["unemployment_ratio"] , data["births_ratio"], kind="kde",size=7)
plt.show()

sns.lmplot(x="unemployment_ratio", y="births_ratio", data=data)
plt.show()

# Lm plot also show us correlation.

# Last tool for correlation is heatmap.
data.corr()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True,ax=ax)
plt.show()
