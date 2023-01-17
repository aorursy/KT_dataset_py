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

import seaborn as sns

import matplotlib.pyplot as plt
from subprocess import check_output
d1school= pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")
d2policekill= pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="windows-1252")
d2policekill.head()
d3income= pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding="windows-1252")

d4racecity= pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding="windows-1252")

d5poverty= pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")
d5poverty.head()
d5poverty.poverty_rate.value_counts(dropna=False)
d5poverty.poverty_rate.replace(["-"],0.0,inplace=True)
d5poverty.poverty_rate.astype(float)
d5poverty.info()
d5poverty.poverty_rate=d5poverty.poverty_rate.astype(float)
d5poverty.dtypes
d5poverty.head()
d5poverty.groupby("Geographic Area")
area_list= list(d5poverty["Geographic Area"].unique())
area_rate=[]

for i in area_list:

    x=d5poverty[d5poverty["Geographic Area"]==i]

    area_rate.append(sum(x.poverty_rate)/len(x))
data=pd.DataFrame({"area_name":area_list,"area_rate":area_rate})
data
data.sort_values(by="area_rate",ascending=False,inplace=True)
data
plt.figure(figsize=(10,12))

axes=sns.barplot(data.area_name,data.area_rate)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,10))

axes=sns.barplot(data.area_name,data.area_rate)

plt.xticks(rotation=90)

plt.xlabel("states")

plt.ylabel("Poverty Rates")

plt.title("Poverty Rates of States")
d2policekill.head()
names_list=list(d2policekill["name"])
names_list
names_list_sep=[]

for i in names_list:

    names_list_sep.append(i.split(" "))
names_list_sep
a,b = zip(*names_list_sep)
names2=a+b
import collections

names_count=collections.Counter(names2)
names_count
most_common=names_count.most_common(15)
most_common
x,y=zip(*most_common)
x,y=list(x),list(y)

x.pop(0)

y.pop(0)
plt.figure(figsize=(10,10))

sns.barplot(x,y,palette=sns.cubehelix_palette(len(x)))
asian=[10,20,30]

white=[5,2,3]

hispanic=[6,3,1]

area=["ist","ank","bur"]

f,ax= plt.subplots(figsize=(4,3))

sns.barplot(asian,area,label="asian",alpha=1,color="yellow")

sns.barplot(white,area,label="white",alpha=1,color="red")

sns.barplot(hispanic,area,label="hispanic",alpha=1,color="blue")

ax.legend(frameon=True)