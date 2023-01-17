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
data=pd.read_csv("../input/tmdb_5000_movies.csv")

data.info()
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidth=.5, fmt=".1f",ax=ax)
plt.show()
data.plot(kind="bar",x="budget",y="vote_count",figsize=(12,12),color="red")
plt.xlabel("budget")
plt.ylabel("vote_count")
plt.show()

data.plot(kind="scatter",x="revenue",y="budget",label="Movies",color="red")
plt.xlabel("revenue")
plt.ylabel("budget")
plt.show()
data.budget.plot(kind="line",figsize=(12,12))
plt.show()
data.budget.plot(kind="hist",grid=True,bins=50)
plt.xlabel("budget")
plt.ylabel("frequance")
plt.show()
data.revenue.plot(kind="line",grid=True,figsize=(12,12))
plt.xlabel("id")
plt.ylabel("revenue")
plt.show()
dictionary={"Bütce":data["budget"],"Dil":data["original_language"],"Adı":data["original_title"]}
print(dictionary)

newDataframe=pd.DataFrame(dictionary)
newDataframe.info()

filter1=data["budget"]>20000000
filter2=data["id"]<100
newdata=data[filter1&filter2]
print(newdata)

print(data.loc[:5,"budget":"original_language"])
i=0
for each in dictionary["Adı"]:
    if (i>15):
        break   
    else:
        print(each)
        i+=1
   
for each in dictionary["Bütce"]:
    if(each>250000000):
         print(each)
