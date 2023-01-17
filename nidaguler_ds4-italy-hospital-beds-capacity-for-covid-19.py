import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_per_italy_v1.csv")
df.head()
df.tail()
df.columns=['country', 'state', 'county', 'lat', 'lon', 'type', 'measure', 'beds',

       'population', 'year', 'source', 'source_url']
df.drop(["country","county","measure","source","source_url"],axis=1,inplace=True)
df.head()
df.isna().sum()
df.info()
data=df.iloc[:,2:4]
df.drop(["lat","lon"],axis=1,inplace=True)
df.sample(5)
df.describe()
df.type.unique()
df.drop(["type"],axis=1,inplace=True)
df.corr()
plt.figure(figsize=(20,7))

sns.barplot(x=df["state"].value_counts().index,

y=df["state"].value_counts().values)

plt.title("state other rate")

plt.ylabel("rates")

plt.legend(loc=0)

plt.xticks(rotation=90)

plt.show()
df.head()
plt.figure(figsize=(20,7))

ax = sns.pointplot(x="state", y="beds", hue="year",data=df)

plt.xticks(rotation=90)

plt.show()