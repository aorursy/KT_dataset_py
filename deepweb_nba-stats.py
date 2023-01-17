# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
player_data = pd.read_csv('../input/nba-players-stats/player_data.csv')
players = pd.read_csv('../input/nba-players-stats/Players.csv')
seasons = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')
players.head(2)
print(players.shape)
players.isnull().sum()
players.info()
players.describe()
# drop the first "unnamed" column
players.drop('Unnamed: 0', axis=1, inplace=True)
players.head(2)
players.dropna(how="all", inplace=True)
players.set_index("Player", inplace=True)
players.head(2)
print("Tallest player : {0} - {1} cm".format(players["height"].idxmax(), players["height"].max()))
print("Smallest player: {0} - {1} cm".format(players["height"].idxmin(), players["height"].min()))
print()
print("Heaviest player: {0} - {1} kg".format(players["weight"].idxmax(), players["weight"].max()))
print("Lightest player: {0} - {1} kg".format(players["weight"].idxmin(), players["weight"].min()))
print()
print("Height average of players: ", players["height"].mean())
print("Weight average of players: ", players["weight"].mean())
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
bins = range(100, 300, 10)
plt.hist(players["height"], bins, histtype="bar")
plt.xlabel("Height (cm)")
plt.ylabel('Count')
plt.axvline(players["height"].mean(), color="r", linestyle='dashed', linewidth=3)
plt.plot()

plt.subplot(1, 2, 2)
bins = range(60,180,10)
plt.hist(players["weight"],bins, histtype="bar", rwidth=1.2, color='#4400ff')
plt.xlabel('Weight in Kg')
plt.ylabel('Count')
plt.axvline(players["weight"].mean(), color='black', linestyle='dashed', linewidth=2)

plt.subplots_adjust(bottom=0.15, wspace=0.40)
plt.plot()
players.groupby(["collage"])["height"].count()
players_grouped_by_collage = players.groupby(["collage"])["collage"]
print(players_grouped_by_collage.head(2))
print("\n\n")
print(players_grouped_by_collage.count())
maximum_player_threshold = 40
height_of_grouped_plyrs = players.groupby(["collage"])["height"]
top_collages = height_of_grouped_plyrs.count()[height_of_grouped_plyrs.count() > maximum_player_threshold]
top_collages = top_collages.reset_index().sort_values(by="height", ascending=False)
top_collages.set_index("collage", inplace=True)
top_collages.head()
# replace the column name "height" with "Count"
top_collages.columns = ["Count"]
top_collages.head()
axis_font = {'size':'24', 'color':'black', 'weight':'bold'}

ax = top_collages.plot.bar(width=0.8, figsize=(20, 10), fontsize=24, colormap="viridis", alpha=0.6)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.35))
    
plt.show()
players.plot(x="height", y="weight", kind="scatter", figsize=(12, 8))
player_data
player_start_end = player_data[["name", "year_end", "year_start"]]
player_start_end["years_played"] = player_data["year_end"] - player_data["year_start"]
max_val_index = player_start_end["years_played"].argmax() # alternative => player_start_end["years_played"].idxmax()
player_start_end.iloc[max_val_index]
players_by_years_played = player_start_end.sort_values(["years_played"], ascending=False)
players_by_years_played[players_by_years_played["years_played"] >= 15]
filtered_players = players_by_years_played[players_by_years_played["years_played"] >= 15]
filtered_players
pd.merge(seasons, filtered_players, how='inner', left_on="Player", right_on="name", sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
print(seasons.shape)
print(player_data.shape)
seasons[seasons["Player"] == "Zach Randolph"]
player_data.shape
