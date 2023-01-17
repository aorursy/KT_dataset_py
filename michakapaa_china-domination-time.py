import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn .preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
# Set path to data 

path = "../input/top-women-chess-players/top_women_chess_players_aug_2020.csv"
#Read data

data = pd.read_csv(path)
data.sample(10)
data.describe()
np.array(data.Inactive_flag != "wi").sum()
data = data[data.Inactive_flag != "wi"]
plt.figure(figsize=(13, 6))

data.Year_of_birth.hist(bins=20)
current_age = 2020

plt.figure(figsize=(13, 6))

data["Age"] = current_age - data.Year_of_birth

data.Age.plot.box()
print("Median of age: ", data.Age.median())
federation_data = data.groupby("Federation").size().head(20)
federation_data = federation_data.sort_values(ascending=False)
plt.figure(figsize=(13, 6))

plt.title("Active players by federation")

plt.ylabel("Count")

federation_data.plot.bar()
data.Title.unique()
title_data = data.groupby("Title")
plt.figure(figsize=(9, 9))

title_data.size().plot.pie()
plt.figure(figsize=(13, 6))

plt.title("Standard rating hist")

plt.xlabel("Rating")

plt.ylabel("Count")

data.Standard_Rating.hist()
plt.figure(figsize=(13, 6))

plt.title("Blitz rating hist")

plt.xlabel("Rating")

plt.ylabel("Count")

data.Blitz_rating.hist()
plt.figure(figsize=(13, 6))

plt.title("Rapid rating hist")

plt.xlabel("Rating")

plt.ylabel("Count")

data.Rapid_rating.hist()
srt_by_federation = data.groupby("Federation").Standard_Rating.mean()

srt_by_federation = srt_by_federation.sort_values(ascending=False)

srt_by_federation = srt_by_federation.head(20)
plt.figure(figsize=(13, 6))

plt.title("Standard rating mean")

plt.ylabel("Rating")

plt.axis([0,0,2000, 2200])

srt_by_federation.plot.bar()
brt_by_federation = data.groupby("Federation").Blitz_rating.mean()

brt_by_federation = brt_by_federation.sort_values(ascending=False)

brt_by_federation = brt_by_federation.head(20)
plt.figure(figsize=(13, 6))

plt.title("Blitz rating mean")

plt.ylabel("Rating")

plt.axis([0,0,1800, 2200])

brt_by_federation.plot.bar()
rrt_by_federation = data.groupby("Federation").Rapid_rating.mean()

rrt_by_federation = rrt_by_federation.sort_values(ascending=False)

rrt_by_federation = rrt_by_federation.head(20)
plt.figure(figsize=(13, 6))

plt.title("Rapid rating mean")

plt.ylabel("Rating")

plt.axis([0,0,1900, 2200])

rrt_by_federation.plot.bar()
srt_max_federation = data.groupby("Federation").Standard_Rating.max()

srt_max_federation = srt_max_federation.sort_values(ascending=False)

srt_max_federation = srt_max_federation.head(20)
plt.figure(figsize=(13, 6))

plt.title("Standard rating max value")

plt.ylabel("Rating")

plt.axis([0,0,2400, 2700])

srt_max_federation.plot.bar()
brt_max_federation = data.groupby("Federation").Blitz_rating.max()

brt_max_federation = brt_max_federation.sort_values(ascending=False)

brt_max_federation = brt_max_federation.head(20)
plt.figure(figsize=(13, 6))

plt.title("Blitz rating max value")

plt.ylabel("Rating")

plt.axis([0,0,2300, 2700])

brt_max_federation.plot.bar()
rrt_max_federation = data.groupby("Federation").Rapid_rating.max()

rrt_max_federation = rrt_max_federation.sort_values(ascending=False)

rrt_max_federation = rrt_max_federation.head(20)
plt.figure(figsize=(13, 6))

plt.title("Rapid rating max value")

plt.ylabel("Rating")

plt.axis([0,0,2300, 2700])

rrt_max_federation.plot.bar()
rating_data = data[["Standard_Rating", "Blitz_rating", "Rapid_rating"]]
rating_data.plot.scatter(x="Blitz_rating", y="Standard_Rating", figsize=(13,6),title=("Standard and blitz rating relation"))
rating_data.plot.scatter(x="Rapid_rating", y="Standard_Rating", figsize=(13,6), title=("Standard and rapid rating relation"))
rating_data.plot.scatter(x="Rapid_rating", y="Blitz_rating", figsize=(13,6), title=("Blitz and rapid rating relation"))
fig = plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')

ax.scatter3D(data.Blitz_rating, data.Rapid_rating, data.Standard_Rating, zdir="z")

ax.set_xlabel('Blitz Rating')

ax.set_ylabel('Rapid Rating')

ax.set_zlabel('Standard Rating');
cm= sns.light_palette("seagreen", as_cmap=True)

sns.heatmap(rating_data.corr(), cmap=cm)
plt.figure(figsize=(13, 6))

plt.title("Title standard rating")

plt.ylabel("Rating")

plt.axis([0,0,1900, 2500])

title_data.Standard_Rating.mean().plot.bar()
data = data.sample(frac=1).reset_index(drop=True)
data.drop(["Fide id", "Name", "Gender", "Inactive_flag", "Year_of_birth"], axis=1, inplace=True)

data.head()
data.info()
data.Blitz_rating.fillna(data.Blitz_rating.mean(), inplace=True)

data.Rapid_rating.fillna(data.Rapid_rating.mean(), inplace=True)

data.dropna(axis=0, inplace=True)
data.Title = [ "Other" if x != 'GM' and x != 'IM' else x for x in data.Title]
lb = LabelEncoder()

title_map = {"GM":2, "IM":1, "Other":0}

data.Title = data.Title.map(title_map)

data.Federation = lb.fit_transform(data.Federation)
data[["Federation", "Title", "Age"]].corr()
data.drop(["Federation","Age"],axis=1, inplace=True)
X = data.drop(["Title"], axis=1)

y = data.Title
sc = StandardScaler()

X = sc.fit_transform(X)
sgd = LogisticRegression()

cross_val_score(sgd, X, y, cv=5)