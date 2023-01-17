import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



%matplotlib inline
data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
data.isnull().sum()
data.shape
for column in data.columns:

    print(f'P{column} {len(data.host_id.unique())}')
data["minimum_nights"].describe()
data["price_per_day"] = data["price"]/data["minimum_nights"]

data["price_per_day"][:5] #Check with dataframe to see if 80 is divide by 10 to get 8
data.head()
data["Neighbour"] = data["neighbourhood"].astype(str)+"_"+data["neighbourhood_group"].astype(str)
sns.barplot(data["neighbourhood_group"].value_counts().index, data["neighbourhood_group"].value_counts().values)

plt.title("Number of Listing Per Neighbourhood Group")

plt.show()
for ng in data["neighbourhood_group"].unique():

    print(f'Avg Price of Neighbourhood Group {ng}: {data[data["neighbourhood_group"]==ng]["price_per_day"].sum()/len(data[data["neighbourhood_group"]==ng])}')
grp_neighbour = data.groupby("neighbourhood_group")["price_per_day"].mean().sort_values()

plt.figure(figsize=(10, 10))

plt.xticks(rotation=45)

plt.xlim(0,max(grp_neighbour.values))

sns.barplot(grp_neighbour.values,grp_neighbour.index)

plt.title("Avg price per day of neighbour group")

plt.show()
plt.figure(figsize=(10, 10))

sns.boxplot(x = 'price_per_day', y = 'neighbourhood_group', data = data)

plt.title("Neighbourhood Group vs Price per day")

plt.show()
data[data["neighbourhood_group"]=="Manhattan"][:5]
data[data["neighbourhood_group"]=="Manhattan"].describe()
data[data["neighbourhood_group"]=="Manhattan"][["price", "price_per_day", "minimum_nights"]].sort_values(by="price")
data[data["neighbourhood_group"]=="Brooklyn"].describe()
data[data["neighbourhood_group"]=="Staten Island"].describe()
data[data["neighbourhood_group"]=="Queens"].describe()
data[data["neighbourhood_group"]=="Bronx"].describe()
neighbour = data.groupby("Neighbour")["price_per_day"].mean().sort_values()
plt.figure(figsize=(10, 10))

plt.xticks(rotation=45)

plt.xlim(0,max(neighbour.values))

sns.barplot(neighbour.values[:25],neighbour.index[:25])

plt.title("Top 25 on Avg Cheapest Neighbourhood + Neighbourhood_group")

plt.show()
neighbour = neighbour.sort_values(ascending=False)

plt.figure(figsize=(10, 10))

plt.xticks(rotation=45)

plt.xlim(0,max(neighbour.values))

sns.barplot(neighbour.values[:25],neighbour.index[:25])

plt.title("Top 25 on Avg Expensive Neighbourhood + Neighbourhood group")

plt.show()
data["room_type"].unique()
data.columns
data_ = data[["room_type", "price_per_day", "number_of_reviews", "neighbourhood_group"]]
data_.head()
data_.groupby(["neighbourhood_group", "room_type"]).count()


plt.figure(figsize=(12,12))

plt.grid()

g = sns.barplot(data_.groupby(["neighbourhood_group", "room_type"])["price_per_day"].count().values, data_.groupby(["neighbourhood_group", "room_type"])["price_per_day"].count().index)

g.set_xscale("log")

plt.title("Room Type Count in each Neighbourhood")

plt.xlabel("Number of Listings Available")

plt.ylabel("Neighbourhood and Room Type")

plt.show()
data_.groupby(["neighbourhood_group", "room_type"])["price_per_day"].mean()
plt.figure(figsize=(14,12))

plt.grid()

sns.barplot(data_.groupby(["neighbourhood_group", "room_type"])["price_per_day"].mean().values, data_.groupby(["neighbourhood_group", "room_type"])["price_per_day"].mean().index)

plt.title("Price Per Day for each Room Type in each Neighbourhood")

plt.xlabel("Price")

plt.ylabel("Neighbourhood and Room Type")

plt.show()