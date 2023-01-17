import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





import os

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_per_sweden_v1.csv")
df.head()
df.tail()
df.drop(["country","county","source","source_url"],axis=1,inplace=True)
df.head()
df.isna().sum()
df.info()
data=df.iloc[0,1:3]
df.drop(["lat","lng"],axis=1,inplace=True)
df.sample(5)
df.describe()
df.type.unique()
sns.set(style="whitegrid")

ax=sns.barplot(x=df["type"].value_counts().index,

              y=df["type"].value_counts().values,palette="Blues_d",

              hue=['ICU', 'TOTAL'])

plt.legend(loc=9)

plt.xlabel("type")

plt.ylabel("Frequency")

plt.title("Show of type Bar Plot")

plt.show()
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

ax = sns.pointplot(x="state", y="beds", hue="type",data=df)

plt.xticks(rotation=90)

plt.show()
labels=df['type'].value_counts().index

colors=['blue','red','yellow','green']

explode=[0.1,0]

values=df['type'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('type According Analysis',color='black',fontsize=10)

plt.show()