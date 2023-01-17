!pip install -q -U seaborn==0.11.0 --use-feature=2020-resolver
import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns





sns.set(font_scale=1.5)
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")

data.head()
data.dtypes
data["salary"] = data["salary"].str[1:].astype("int64")

data["height"] = data["height"].str.split("/").str[1].astype("float")

data["weight"] = data["weight"].str.split("/").str[1].str[0:-3].astype("float")

data["start_age"] = data["draft_year"] - pd.to_datetime(data["b_day"]).dt.year

data["draft_round"] = data["draft_round"].replace({"Undrafted": 0}).astype("int8")

data["draft_peak"] = data["draft_peak"].replace({"Undrafted": 0}).astype("int8")
plt.figure(figsize=(15, 5))

plt.xlabel("Rating", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title("Rating distribution", fontsize=18)

sns.countplot(x="rating", data=data, palette="rocket");
plt.figure(figsize=(15, 5))

plt.xlabel("Rating", fontsize=14)

plt.ylabel("Salary", fontsize=14)

plt.title("Salary dependence on player rating", fontsize=18)

sns.scatterplot(x="rating", y="salary", data=data, color="deeppink", s=60);
plt.figure(figsize=(15, 5))

plt.xlabel("Draft year", fontsize=14)

plt.ylabel("Salary", fontsize=14)

plt.title("Salary dependence on draft year", fontsize=18)

sns.scatterplot(x="draft_year", y="salary", data=data, color="deeppink", s=60);
plt.figure(figsize=(15, 5))

plt.xlabel("Position", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title("Position distribution", fontsize=18)

sns.countplot(x="position", data=data, palette="rocket");
plt.figure(figsize=(15, 8))

plt.xlabel("Position", fontsize=14)

plt.ylabel("Salary", fontsize=14)

plt.title("Salary distribution based on players positions", fontsize=18)

sns.boxplot(x="position", y="salary", data=data, palette="rocket");
plt.figure(figsize=(15, 8))

plt.xlabel("Position", fontsize=14)

plt.ylabel("Height", fontsize=14)

plt.title("Height distribution based on players positions", fontsize=18)

sns.boxplot(x="position", y="height", data=data, palette="rocket");
plt.figure(figsize=(15, 5))

plt.xlabel("Start age", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title("Age distribution", fontsize=18)

sns.histplot(x="start_age", data=data, bins=20, color="deeppink");
plt.figure(figsize=(15, 8))



plt.title("Salary distribution based on players countries", fontsize=18)

x = sns.boxplot(x="country", y="salary", data=data, color="deeppink")

x.set_xticklabels(x.get_xticklabels(), rotation=90);
plt.figure(figsize=(15, 8))

plt.title("Salary distribution based on players teams", fontsize=18)

x = sns.boxplot(x="team", y="salary", data=data, color="deeppink")

x.set_xticklabels(x.get_xticklabels(), rotation=90);
colleges = data.groupby("college")["salary"].median()

top_colleges = colleges.sort_values(ascending=False)[0:10].index

tmp = data[data['college'].isin(top_colleges)]



plt.figure(figsize=(10,5))

x = sns.barplot(x="college", y="salary", data=tmp, color="deeppink")

x.set_xticklabels(x.get_xticklabels(), rotation=90);
plt.figure(figsize=(15,10))



sns.heatmap(data.corr(),cmap='rocket',annot=True, fmt=".2f");