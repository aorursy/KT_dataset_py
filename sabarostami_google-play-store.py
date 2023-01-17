# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/googleplaystore.csv")

df.head()
df.info()
df["App"].unique()
len(df["App"].unique())
len(df["Category"].unique())
df["App"].groupby(df["Category"]).count()
df.shape
idf = df[df["Category"] == "1.9"]

idf
df = df.drop(index=10472)

df.shape
cdf = df.drop_duplicates()

cdf.shape
df.shape[0] - cdf.shape[0]
cdf = cdf.drop_duplicates(subset="App")
cdf.shape
cdf.isnull().sum()
cdf.Reviews = pd.to_numeric(cdf.Reviews)

cdf.Rating = pd.to_numeric(cdf.Rating)
def size_converter(col):

    if col[-1] == "M":

        return float(col[:-1])

    elif col[-1] == "K":

        return float(col[:-1]) / 1024

    else:

        return 0

    

cdf["Size"] = cdf["Size"].apply(size_converter)
cdf.Size[:10]
cdf.Size.describe()
cdf[cdf["Size"] == 0.0].shape
cdf[cdf["Size"] == 100].groupby(cdf["Category"])["Size"].count().sort_values(ascending=False)
cdf.Installs[:10]
def installs(col):

    x = col.split("+")[0]

    try:

        if int(x) <= 100:

            return "100-"

        else:

            return "100+"

    except ValueError:

        return col

    

    

cdf.Installs = cdf.Installs.apply(installs)

cdf.Installs[:10]
cdf.Type.unique()
cdf[cdf["Type"].isnull()]
cdf.drop(index=9148, inplace=True)
cdf.shape
cdf.Price.unique()
cdf["Price"].describe()
def price_converter(col):

    try:

        x = col.split("$")[1]

        return float(x)

    except IndexError:

        return float(col)

cdf.Price = cdf.Price.apply(price_converter)

cdf.Price[cdf["Price"] > 0].sort_values()[:10]
cdf.Price[cdf["Price"] > 100].sort_values()[:10]
cdf["Price"].max()
cdf[cdf["Price"] == 400.0]
cdf.columns
cdf["Content Rating"].unique()
cdf["Genres"].nunique()
cdf[["Category", "Genres"]][:50]
cdf.drop("Genres",axis=1, inplace=True)
cdf["Last Updated"][:20]
cdf["Last Updated"] = pd.to_datetime(cdf["Last Updated"])

cdf["Last Updated"][:10]
cdf["Current Ver"].nunique()
cdf["Android Ver"].nunique()
cdf.drop(["Current Ver", "Android Ver"], axis=1, inplace=True)
cdf.info()
cdf.head()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10,8))

plt.hist(cdf["Rating"])

plt.xlabel("Rating")

plt.show()
plt.figure(figsize=(10,8))

plt.bar(cdf["Category"], cdf["Rating"])

plt.xticks(rotation="vertical")

plt.show()
sns.boxplot(x="Category", y="Rating", data=cdf)

plt.tight_layout()

plt.xticks(rotation="vertical")

plt.axhline(cdf["Rating"].median(), color="red")

plt.show()
fig, axes=plt.subplots(1,2)



cdf[cdf["Category"] == "ART_AND_DESIGN"]["Rating"].plot(kind="hist", ax=axes[0])

cdf[cdf["Category"] == "TOOLS"]["Rating"].plot(kind="hist", ax=axes[1])



axes[0].title.set_text("Art And Design")

axes[1].title.set_text("Tools")
cdf[cdf["Category"] == "ART_AND_DESIGN"]["App"][:10]
cdf[cdf["Category"] == "TOOLS"]["App"][:10]
cdf.groupby("Category")["Rating"].mean().sort_values(ascending=False)
cdf.groupby("Category")["Rating"].mean().sort_values(ascending=True)
cdf["Last Updated"].describe()
cdf[cdf["Last Updated"] == "2010-05-21 00:00:00"]
cdf[cdf["Last Updated"] < "2015-05-21 00:00:00"].shape
print(cdf.shape[0])

print(cdf[cdf["Price"] == 0.0].shape[0])

print(cdf.shape[0] - cdf[cdf["Price"] == 0.0].shape[0])
plt.figure(figsize=(10,10))

X = cdf[cdf["Price"] > 0.0]["Category"]

sns.boxplot(x=X, y="Rating", data=cdf)

plt.xticks(rotation="vertical")

plt.axhline(cdf["Rating"].median(), color="red")

plt.show()
cdf[cdf["Price"] > 0.0].groupby("Category")["Rating"].mean()
cdf.corr()


sns.heatmap(cdf.corr(), cmap="coolwarm")