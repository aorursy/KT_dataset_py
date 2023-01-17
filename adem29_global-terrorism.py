# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/gtd/globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1")
list(data)
data.country_txt.nunique()#How manny countries has been subjected to terror attack.
data.iyear.hist(bins = 100, grid = False,label = "Years")

plt.legend()

plt.show()
data.country_txt.value_counts()[:10]
plt.pie(data.country_txt.value_counts()[0:10], labels = data.country_txt.value_counts()[:10].index)

plt.show()
(round(100*data.country_txt.value_counts()[0:10]/data.country_txt.value_counts().sum(),2))
data[data.nkill == data.nkill.max()]
data.groupby("country_txt").sum()["nkill"].sort_values(ascending = False).tail(10)
data.groupby("country_txt").sum()        #["nkill"]
list(zip(data.iyear,data.country_txt))
plt.plot(data.groupby("iyear").count()["eventid"].index, data.groupby("iyear").count()["eventid"], color ="r",label = "Number of T. Attacks")

plt.plot(data.groupby("iyear").sum()["nkill"].index, data.groupby("iyear").sum()["nkill"], color ="blue", label = "Number of Murdered")

plt.legend()

plt.show()

data.groupby("iyear").count()["eventid"].corr (data.groupby("iyear").sum()["nkill"])
data[data.suicide == 1].suicide.count()
data.attacktype1_txt.value_counts()[:10]

plt.bar(data.attacktype1_txt.value_counts().index [:10],data.attacktype1_txt.value_counts() ,color = "blue")

plt.bar(data.attacktype1_txt.value_counts().index [:10],data.attacktype1_txt.value_counts())[3].set_color("red")

plt.xticks(rotation = '90')
data.attacktype1_txt.value_counts()
data.groupby("attacktype1_txt").sum().loc[data.attacktype1_txt.value_counts().index]["nkill"]
a = data.groupby("attacktype1_txt").sum().loc[data.attacktype1_txt.value_counts().index]["nkill"]/data.attacktype1_txt.value_counts()

print(a)
plt.bar(a.index,a)

plt.xticks(rotation = 90)

plt.show()
list1 = []

for i in range(5):

    for j in range(1970,2018):

        a = data[(data.country_txt == data.country_txt.value_counts().index[:10][i]) & (data.iyear == j)].nkill.sum()

        list1.append(a)

    plt.plot(range(1970,2018), list1, label = data.country_txt.value_counts().index[:10][i])

    plt.legend()

    list1 = []
data.nkill[(data.country_txt == "Iraq") & (data.iyear == 2014)].sum()