import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv", engine='python')
df.head()
df.tail()
df.drop(["id"],axis=1,inplace=True)
df.info()
df.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),annot=True,linewidths=.5, fmt='.1f',ax=ax)

plt.show()
df.columns
df.isna().sum()
data=pd.DataFrame(df.loc[::,["age","gender","race","city"]])
data.head()
data.tail()
data.gender.unique()
data=data.dropna()
data.gender.unique()
sns.countplot(data["gender"])

plt.show()

print(data.gender.value_counts())
sns.countplot(data["race"])

plt.show()

print(data.race.value_counts())
data["gender"]=[1 if i.strip()== "M" else 0 for i in data.gender]
print(len(data))
print("Data Shape:", data.shape)
data.info()
data.describe()