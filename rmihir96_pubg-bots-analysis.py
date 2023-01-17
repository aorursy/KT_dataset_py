import pandas as pd

import numpy as np
kills = pd.read_csv("../input/deaths/kill_match_stats_final_0.csv")

matches = pd.read_csv("../input/aggregate/agg_match_stats_0.csv")
matches.shape

matches.head()
kills.head()

kills.shape
#Lets just start with solo games. So we remove other games.

match_stats = matches[matches["party_size"]==1].drop_duplicates()
#Making a list of all the games we are interested in.

match_list = match_stats["match_id"].values

match_zero_kills = match_stats[match_stats["player_kills"]==0]

match_zero_kills.shape
#Heres how our new dataframe with players only with zero kills looks like

match_zero_kills.head()
#Now since the data is too huge, lets just select a match randomly and analyze it further

#It contains all playrs from the game.

selected_match = matches[matches["match_id"]==match_list[4]]

selected_match.shape
#Now since the data is too huge, lets just select a match randomly and analyze it further. 

#This however contains only players with zero kills

selected_game_zero = match_zero_kills[match_zero_kills["match_id"]== match_list[4]]

selected_game_zero.shape
selected_game_zero.head()
#Lets take a look at the kill stats from the same selected game.

kills_in_selected_game = kills[kills["match_id"]==match_list[4]]

kills_in_selected_game.shape
#Lets merge the kill stats and the match stats for the select game.

select_game_dataset = selected_game_zero.merge(kills_in_selected_game, left_on="player_name", right_on="victim_name", how="inner")

select_game_dataset.shape
#This contains all the players, not just the ones with zero kills.



selected_game_dataset_all = selected_match.merge(kills_in_selected_game, left_on="player_name", right_on="victim_name", how="inner")

selected_game_dataset_all.shape
#This is quite intuitive space. A "bot" would never drive a car or survive for a long time, cause they usually get killed as soon as they spawn.

#Dropping players who have driven a vehicle

select_game_dataset = select_game_dataset[select_game_dataset["player_dist_ride"] == 0]

select_game_dataset = select_game_dataset[select_game_dataset["player_survive_time"] < 300]

select_game_dataset.shape
select_game_dataset.head()
features = ["match_id_x", "player_dist_ride", "player_dist_walk", "player_dmg","player_survive_time","team_id", "killed_by","killer_name", "killer_placement", 

            "killer_position_x", "killer_position_y", "time", "victim_name", "victim_placement", "victim_position_x",

            "victim_position_y"]

len(features)
selected_game_bots = select_game_dataset[features]

selected_game_bots.shape
#removed_features = ["Down and Out" , 'Bluezone' , 'death.RedZoneBomb_C' , 'Falling' , 'Drown' , 'RedZone']

selected_game_bots = selected_game_bots[~selected_game_bots.killed_by.str.contains("Down and Out" , 'Bluezone')]

selected_game_bots = selected_game_bots[~selected_game_bots.killed_by.str.contains('death.RedZoneBomb_C' , 'Falling')]

selected_game_bots = selected_game_bots[~selected_game_bots.killed_by.str.contains('Drown' , 'RedZone')]



selected_game_bots.shape
#This is how the "bots" were killed.

selected_game_bots.killed_by.value_counts()
import matplotlib.pyplot as plt

from scipy.misc import imread

img = imread("../input/erangel.jpg")

implot = plt.imshow(img,aspect='auto', extent=[0, 600000, 0, 700000])



# put a blue dot at (10, 20)

plt.plot(selected_game_bots["victim_position_x"], selected_game_bots["victim_position_y"], "o")

#plt.plot(all_dataset["victim_position_x"], all_dataset["victim_position_y"],  "ro")





plt.show()
selected_game_dataset_all = selected_game_dataset_all[features]

selected_game_dataset_all.shape
import matplotlib.pyplot as plt

img = imread("../input/erangel.jpg")

implot = plt.imshow(img,aspect='auto', extent=[0, 600000, 0, 700000])



# put a blue dot at (10, 20)

plt.plot(selected_game_dataset_all["victim_position_x"], selected_game_dataset_all["victim_position_y"], "o")

#plt.plot(all_dataset["killer_position_x"], all_dataset["killer_position_y"], "ro")





plt.show()


img = imread("../input/erangel.jpg")

implot = plt.imshow(img,aspect='auto', extent=[0, 600000, 0, 700000])



# put a blue dot at (10, 20)

#plt.plot(all_dataset["victim_position_x"], all_dataset["victim_position_y"], "o")

plt.plot(selected_game_dataset_all["killer_position_x"], selected_game_dataset_all["killer_position_y"], "ro")





plt.show()