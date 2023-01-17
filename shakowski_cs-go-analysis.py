# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
d_picks = pd.read_csv("/kaggle/input/csgo-professional-matches/picks.csv")
d_economy = pd.read_csv("/kaggle/input/csgo-professional-matches/economy.csv")
d_results = pd.read_csv("/kaggle/input/csgo-professional-matches/results.csv")
d_players = pd.read_csv("/kaggle/input/csgo-professional-matches/players.csv")
d_picks.rename(columns={"team_1":"team_1_pick","team_2":"team_2_pick" }, inplace=True)

new_df = pd.merge(d_results, d_picks, left_on="match_id", right_on="match_id")

#filter for best of 3's
new_df = new_df[new_df["best_of"] == "3"]

#add new column for winning team name
new_df["winning_team_name"] = ""
new_df.loc[new_df["map_winner"]==1, "winning_team_name"] = new_df["team_1"]
new_df.loc[new_df["map_winner"]==2, "winning_team_name"] = new_df["team_2"]

#create column for name of team who picked the map
new_df["map_picker"] = ""
new_df.loc[new_df["_map"] == new_df["t1_picked_1"], "map_picker"] = new_df["team_1_pick"]
new_df.loc[new_df["_map"] == new_df["t2_picked_1"], "map_picker"] = new_df["team_2_pick"]
new_df.loc[new_df["_map"] == new_df["left_over"], "map_picker"] = "left_over"

#remove decider games
new_df = new_df[new_df["map_picker"] != "left_over"]

#create column for pick win
new_df["pick_win"] = new_df["winning_team_name"] == new_df["map_picker"]

#create column for rank of winning team
new_df["winning_rank"] = 0
new_df.loc[new_df["map_winner"]==1, "winning_rank"] = new_df["rank_1"]
new_df.loc[new_df["map_winner"]==2, "winning_rank"] = new_df["rank_2"]

#create column for rank of losing team
new_df["losing_rank"] = 0
new_df.loc[new_df["map_winner"]==1, "losing_rank"] = new_df["rank_2"]
new_df.loc[new_df["map_winner"]==2, "losing_rank"] = new_df["rank_1"]

#filter for only teams in the top twenty
new_df = new_df[new_df['winning_rank']< 21]
new_df = new_df[new_df['losing_rank']< 21]

#filter for after a certain date
new_df['date_x'] = pd.to_datetime(new_df['date_x'])
new_df = new_df[new_df['date_x'] > "12-31-2016"]

#groupby team
grouper = new_df.groupby("map_picker")["pick_win"].value_counts(normalize=True)
grouper = grouper[grouper.index.isin([True], level=1)]

#Remove redundant index and sort
grouper = grouper.reset_index(level=1, drop=True)
grouper = grouper.sort_values(ascending = False)
grouper = grouper.to_frame()
grouper = grouper.reset_index()

#Add column based on number of games
filter_series = new_df["map_picker"].value_counts()
filter_series = filter_series.to_frame()
filter_series = filter_series.reset_index()
grouper = pd.merge(grouper, filter_series, left_on="map_picker", right_on="index")
grouper = grouper.drop("index", 1)
grouper.rename(columns={"map_picker_x":"Team","pick_win": "Win Percentage", "map_picker_y":"Number of Games" }, inplace=True)
grouper = grouper[grouper["Number of Games"] > 20]
print(grouper)