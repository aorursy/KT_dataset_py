# Setup

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



file = "../input/microsoft-common-apps/common_apps.csv"

df = pd.read_csv(file, index_col="App_Order")
df.head()
df.shape
df.describe()
df.dtypes
df.isnull().sum()
df.loc[df["App_Star"].isnull()]
df.loc[df.duplicated()].shape
df = df.rename(columns={"App_Name": "Name", "App_Star": "Stars", "App_Price": "Price ($)","App_Review": "Views"})
df = df.dropna().drop_duplicates(subset="Name")
df["Price ($)"] = df["Price ($)"].replace("Free", "0")

df["Price ($)"] = df["Price ($)"].str.lstrip("$")

df["Price ($)"] = pd.to_numeric(df["Price ($)"])
df["Views"] = df["Views"].astype(int)
df = df.set_index("Name")
df.head()
df.describe()
df.dtypes
df.isnull().sum()
top10 = df.sort_values(by="Views", ascending=False).head(10)

top10
plt.figure(figsize=(10,6))

sns.barplot(x = top10.index, y=top10["Views"]/1000)

plt.title("Most viewed apps", fontsize=16)

plt.xlabel("App", fontsize=12)

plt.xticks(rotation=90)

plt.ylabel("Number of views in thousands", fontsize=12)
top10_5star = df.sort_values(by=["Stars","Views"], ascending=False).head(10)

top10_5star
plt.figure(figsize=(10,6))

sns.barplot(x = top10_5star.index, y=top10_5star["Views"]/1000)

plt.title("Most viewed 5-Star apps", fontsize=16)

plt.xlabel("App", fontsize=12)

plt.xticks(rotation=90)

plt.ylabel("Number of views in thousands", fontsize=12)
top10_non_free = df.loc[df["Price ($)"] > 0].sort_values(by=["Views"], ascending=False).head(10)

top10_non_free
plt.figure(figsize=(10,6))

sns.barplot(x = top10_non_free.index, y=top10_non_free["Views"]/1000)

plt.title("Most viewed non_free apps", fontsize=16)

plt.xlabel("App", fontsize=12)

plt.xticks(rotation=90)

plt.ylabel("Number of views in thousands", fontsize=12)
df["Stars"].value_counts()
plt.figure(figsize=(10,6))

sns.barplot(x=df["Stars"].value_counts().index, y=df["Stars"].value_counts().values, palette="Blues")

plt.title("Star rating of apps", fontsize=16)

plt.xlabel("Stars", fontsize=12)

plt.ylabel("Number of ratings", fontsize=12)
free_apps = df.loc[df["Price ($)"] == 0]

non_free_apps = df.loc[df["Price ($)"] > 0]
plt.figure(figsize=(10,6))

sns.barplot(x=free_apps["Stars"].value_counts().index, y=free_apps["Stars"].value_counts().values, palette="Blues")

plt.title("Star rating of free apps", fontsize=16)

plt.xlabel("Stars", fontsize=12)

plt.ylabel("Number of ratings", fontsize=12)
plt.figure(figsize=(10,6))

sns.barplot(x=non_free_apps["Stars"].value_counts().index, y=non_free_apps["Stars"].value_counts().values, palette="Blues")

plt.title("Star rating of non-free apps", fontsize=16)

plt.xlabel("Stars", fontsize=12)

plt.ylabel("Number of ratings", fontsize=12)
free_apps.describe()
non_free_apps.describe()
free_apps["Views"].sum()
non_free_apps["Views"].sum()
plt.figure(figsize=(10,6))

sns.regplot(x=df["Stars"], y=df["Views"])
plt.figure(figsize=(10,6))

sns.regplot(x=df["Stars"], y=df["Views"])

plt.gca().set_ylim(0, 50000)
sns.lmplot(x="Stars", y="Views", hue="Price ($)", data=df)

plt.gca().set_ylim(0, 50000)