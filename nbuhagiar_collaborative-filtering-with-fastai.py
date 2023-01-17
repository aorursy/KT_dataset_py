# General Data Science

import numpy as np

import pandas as pd

import plotly.express as px



# Collaborative Filtering

from fastai.collab import CollabDataBunch, collab_learner



# Miscellaneous

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
rating = pd.read_csv("/kaggle/input/anime-recommendations-database/rating.csv")

anime = pd.read_csv("/kaggle/input/anime-recommendations-database/anime.csv")

provided_data = rating.merge(anime, how="left", on="anime_id", suffixes=("", "_overall"))

provided_data
for field in ["user_id", "anime_id"]:

    provided_data[field] = provided_data[field].astype(str)

provided_data

provided_data.describe(include="all")
provided_data.groupby(["anime_id", "name"]).size().reset_index().describe(include="all")
provided_data.loc[provided_data["name"] == "Saru Kani Gassen"]
provided_data.drop("anime_id", axis=1, inplace=True)
print("Number of unique users:", len(provided_data["user_id"].unique()))

print("Number of unique anime:", len(provided_data["name"].unique()))
num_user_views = (provided_data["user_id"].value_counts()

                  .reset_index()

                  .rename(columns={"index": "User", "user_id": "Number of Views by User"}))

fig = px.histogram(num_user_views, x="Number of Views by User")

fig.show()
num_anime_views = (provided_data["name"].value_counts()

                   .reset_index()

                   .rename(columns={"index": "Anime", "name": "Number of Views of Anime"}))

fig = px.histogram(num_anime_views, x="Number of Views of Anime")

fig.show()
rating_distro = (provided_data["rating"].value_counts()

                 .sort_index()

                 .reset_index()

                 .rename(columns={"index": "Rating", "rating": "Frequency"}))

fig = px.bar(rating_distro, x="Rating", y="Frequency")

fig.show()
most_popular_animes = (provided_data["name"].value_counts()[:10]

                       .reset_index()[::-1]

                       .rename(columns={"index": "Anime", "name": "View Frequency"}))

fig = px.bar(most_popular_animes, x="View Frequency", y="Anime", orientation="h")

fig.show()
data_filter = provided_data["rating"] != -1

provided_data.drop(["genre", "type", "episodes", "rating_overall", "members"], axis=1, inplace=True)

train_val = provided_data[data_filter]

test = provided_data[~data_filter]
data = CollabDataBunch.from_df(train_val, user_name="user_id", item_name="name", rating_name="rating", seed=0, test=test, bs=256)

data
data.show_batch()
learn = collab_learner(data, n_factors=100, y_range=[1, 10])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 3e-1)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 3e-3)
most_popular_animes_bias = (pd.Series({anime: bias for anime, bias in zip(most_popular_animes["Anime"],

                                                                          learn.bias(most_popular_animes["Anime"]).tolist())})

                            .reset_index()

                            .rename(columns={"index": "Anime", 0: "Bias"}))

fig = px.bar(most_popular_animes_bias, x="Bias", y="Anime", orientation="h")

fig.show()