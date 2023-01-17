# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import codecs
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1", low_memory=False)

data.info()
data.describe()
data.head(3)
data.country_txt = [each.lower() for each in data.country_txt]
data.country_txt = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.country_txt]
data.info("columns")
dataFrame=data[["iyear","imonth","iday","country_txt","region_txt","city",
                "attacktype1","attacktype1_txt","targtype1","targtype1_txt",
               "targsubtype1","targsubtype1_txt","weaptype1","weaptype1_txt","nkill","nwound"]]
dataFrame.head(5)
usdata = dataFrame[dataFrame.country_txt == "united_states"]
ukdata = dataFrame[dataFrame.country_txt == "united_kingdom"]
f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(dataFrame.corr(), annot=False, linewidths=.5, fmt =".1f=", ax=ax)
plt.show()
plt.subplots(figsize=(15,7))
sns.countplot("iyear", data=ukdata, palette="Blues", 
              edgecolor=sns.color_palette("dark",15), label="UK-Blue")
sns.countplot("iyear", data=usdata, palette="Greens", 
              edgecolor=sns.color_palette("dark",15), label="USA_Green")
plt.xticks(rotation=90)
plt.ylabel("Value")
plt.legend()
plt.title("UK&USA Terror attacks, each year")
plt.show()
plt.subplots(figsize=(14,7))
plt.plot(ukdata.nkill, ukdata.nwound, color="green", alpha=1, linestyle=":",label="UK")
plt.plot(usdata.nkill, usdata.nwound, color="blue", alpha=1, linestyle=":", label="USA")
plt.xlabel("nwound")
plt.ylabel("nkill")
plt.title("UK & USA / Number of death and wounded")
plt.legend()
plt.show()
plt.bar(usdata.attacktype1_txt, usdata.attacktype1, color="blue", label="USA")
plt.xticks(rotation=90)
plt.xlabel("attacktype_txt")
plt.ylabel("attacktype1")
plt.legend()
plt.title("Attack Type")
plt.show()
plt.bar(ukdata.attacktype1_txt, ukdata.attacktype1, color="yellow", label="UK")
plt.xticks(rotation=90)
plt.xlabel("attacktype_txt")
plt.ylabel("attacktype1/Value")
plt.legend()
plt.title("Attack Type")
plt.show()
fig_size=plt.rcParams["figure.figsize"]
print("Current Size:" , fig_size)
fig_size[0]=14
fig_size[1]=7
plt.rcParams["figure.figsize"] = fig_size
plt.scatter(ukdata.weaptype1_txt, ukdata.targtype1_txt, color="black", label="UK")
plt.scatter(usdata.weaptype1_txt, usdata.targtype1_txt, color="orange", label="USA")
plt.xticks(rotation=90)
plt.xlabel("weapon type")
plt.ylabel("target type")
plt.title("Weapon  Types & Targets")
plt.legend()
plt.show()