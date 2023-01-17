import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as patches

%matplotlib inline
episodes = pd.read_csv("../input/simpsons_episodes.csv")

locations = pd.read_csv("../input/simpsons_locations.csv")
episodes.head()
mean_episode_ratings = episodes.groupby('season')['imdb_rating'].mean()

ind = np.arange(len(mean_episode_ratings))

labels = ["Season {season}".format(season = season) for season in mean_episode_ratings.index]

plt.bar(ind, mean_episode_ratings, color=(.3,.23,.42))

plt.gca().set_xticks(ind + .4)

plt.gca().set_xticklabels(labels)

plt.ylabel("IMDB Rating")

plt.xticks(rotation=90)

plt.show()
mean_viewership = episodes.groupby('season')['us_viewers_in_millions'].mean()

ind = np.arange(len(mean_episode_ratings))

labels = ["Season {season}".format(season = season) for season in mean_viewership.index]

plt.bar(ind, mean_viewership, color=(.1,.83,.22))

plt.gca().set_xticks(ind + .4)

plt.gca().set_xticklabels(labels)

plt.ylabel("US Viewership in Millions")

plt.xticks(rotation=90)

plt.show()