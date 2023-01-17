

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_per_india_v1.csv")
df.head()
df.tail()
df.drop(["country","county","source","source_url"],axis=1,inplace=True)
df.head()
data=df.iloc[0,1:3]
df.drop(["lat","lng"],axis=1,inplace=True)
df.head()
df.isna().sum()
df.info()
df.sample(5)
df.describe()
df.type.unique()
plt.figure(figsize=(20,7))

sns.barplot(x=df["state"].value_counts().index,

y=df["state"].value_counts().values)

plt.title("state other rate")

plt.ylabel("rates")

plt.legend(loc=0)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(22,7))

sns.barplot(x = "state", y = "beds", hue = "type", data = df)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.pointplot(x="year", y="beds", hue="type",data=df)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.pointplot(x="state", y="beds", hue="type",data=df)

plt.xticks(rotation=90)

plt.show()