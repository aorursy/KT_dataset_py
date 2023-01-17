# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

data.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.figure(figsize=(20,15))

sns.countplot(data=data,x="Platform")
plt.figure(figsize=(15,10))

sns.countplot(data=data,x="Genre")
data.sort_values(by="Global_Sales",ascending=False)[["Name","Global_Sales"]][0:20]
data.groupby(by="Name").sum()["Global_Sales"].sort_values(ascending=False)[0:21]
data.groupby(by="Name").sum()["Global_Sales"].sort_values(ascending=False)[21:40]
data.groupby(by="Name").sum()["Global_Sales"].sort_values(ascending=False)[40:51]
maxlist=[]

for i in list(data.groupby(by="Platform")[["Global_Sales"]].idxmax().values):

    maxlist.append(int(i))

data.loc[maxlist][["Platform","Name","Global_Sales"]]
data.groupby(by="Genre")["Global_Sales"].sum().sort_values(ascending=False)
data.groupby(by=["Genre","Year"])["Global_Sales"].sum().sort_values(ascending=False)[0:25]
data.groupby(by="Year")["Global_Sales"].sum().sort_values(ascending=False)
data.groupby(by="Publisher").count().sort_values(by="Name",ascending=False)["Rank"][0:25]
data.groupby(by="Publisher")["Global_Sales"].sum().sort_values(ascending=False)[0:25]
data[data["Name"].str.contains("Pro Ev")].groupby(by="Name").sum()
data[data["Name"].str.contains("NBA")].groupby(by="Name").sum().sort_values(by="Global_Sales"

,ascending=False).iloc[:,2:][0:20]
data[data["Name"].str.contains("Need")].groupby(by="Name").sum().sort_values(by="Global_Sales"

,ascending=False).iloc[:,2:][0:20]
maxlist=[]

for i in list(data.groupby(by="Genre")[["Global_Sales"]].idxmax().values):

    maxlist.append(int(i))

data.loc[maxlist][["Genre","Name","Global_Sales"]]
data[data["Genre"]=="Racing"].groupby(by="Name").sum().sort_values(by="Global_Sales",ascending=False)[0:10]
for genre in set(data["Genre"]):

    print(genre,": ",end="")

    temp=data[data["Genre"]==genre].groupby(by="Name").sum().sort_values(by="Global_Sales"

             ,ascending=False)[["Global_Sales"]][0:1]

    print(temp[0:1].index[0],end=" ")

    print(temp[0:1].values[0][0])
data[(data["Name"].str.contains("Age of Emp"))|(data["Name"].str.contains("Age of Myt"))].sort_values(

    by="Global_Sales",ascending=False)
data[(data["Genre"]=="Strategy")&(data["Platform"]=="PC")].sort_values(by="Global_Sales"

                                                  ,ascending=False)[["Name","Year","Global_Sales"]][0:20]