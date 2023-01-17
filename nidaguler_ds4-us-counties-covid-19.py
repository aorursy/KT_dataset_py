import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")
df.head()
df.columns=["date","country","state","fip","case","death"]
df.tail()
df.isna().sum()
df.info()
df.describe()
df=df.dropna()
df.sample(5)
df.corr()
plt.figure(figsize=(20,7))

sns.barplot(x=df["country"][:600].value_counts().index,

           y=df["country"][:600].value_counts().values)

plt.xlabel("Country")

plt.ylabel("Frequency")

plt.title("Show of Country Bar Plot")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,7))

sns.barplot(x=df["state"].value_counts().index,

           y=df["state"].value_counts().values)

plt.xlabel("State")

plt.ylabel("Frequency")

plt.title("Show of State Bar Plot")

plt.xticks(rotation=90)

plt.show()