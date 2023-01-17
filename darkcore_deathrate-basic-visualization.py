# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
warnings.filterwarnings("ignore")

data = pd.read_csv("../input/countries of the world.csv")
data.info()
data.head(10)
data.tail()
data.isnull().sum()
plt.figure(figsize=(15,8))
sns.heatmap(data.isnull(),annot=False,cbar=False)
plt.show()
data.dropna(how="any",inplace=True)
plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,fmt="0.1f")
plt.show()
data.columns
data["Region"].unique()
data["Region"].value_counts()
data[data["Population"] > 30000000]
plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot (x=data["Region"].value_counts(),y=data["Region"].unique())
plt.tight_layout()
plt.title("Region Value Counts")
plt.ylabel("Regions")
plt.xlabel("Counts")
plt.show()
regionss = list(data["Region"].unique())
regions = list()
regionss.sort()
for i in regionss:
    i = i.strip()
    regions.append(i)
regions
#object to float64 type

def second(x):
    y = x.split(",")
    if len(y) == 1:
        a = y[0]
    elif len(y) != 1:
        a = y[0] + "." + y[1]
    return a
data["Deathrate"].head()
data["Deathrate"] = data["Deathrate"].apply(second)
data["Deathrate"].head()
data["Deathrate"] = pd.to_numeric(data["Deathrate"])
data["Deathrate"].head(1)
datamelt1 = data.melt(value_vars="Deathrate",value_name="Deathrate",id_vars="Country")
datamelt1["Deathrate"].head()
plt.figure(figsize=(15,30))
sns.set()
sns.barplot(x=datamelt1["Deathrate"],y=datamelt1["Country"].unique())
plt.xlabel("Deathrate")
plt.ylabel("Country")
plt.title("Deathrate-Country")
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,8))
sns.set_style("whitegrid")
region = data.groupby("Region")["Deathrate"]
region.plot()
plt.legend()
plt.tight_layout()
reg = data.groupby("Region")
reg["Deathrate"].sum()
reg["Deathrate"].sum().plot()
plt.show()
regionss = list(data["Region"].unique())
regions = list()
regionss.sort()
for i in regionss:
    i = i.strip()
    regions.append(i)
regions
plt.figure(figsize=(15,10))
sns.barplot(reg["Deathrate"].sum(),regions,palette="autumn")
plt.title("Region-Deathrate (Sum)")
plt.ylabel("Regions")
plt.xlabel("Deathrate (Sum)")
plt.tight_layout()
plt.show()

