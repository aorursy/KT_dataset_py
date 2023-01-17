import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random as r

import warnings

warnings.filterwarnings('ignore')
kills = pd.read_csv('../input/pubg-match-deaths/deaths/kill_match_stats_final_0.csv')
kills = kills.dropna(axis = 0, how = "any") #we will drop all rows with NA values
kills.head()
kills.info()
kills["kill_distance"] = ((kills["killer_position_x"] - kills["victim_position_x"]) ** 2 + (kills["killer_position_y"] - kills["victim_position_y"]) ** 2) ** (1/2)
kills["place_diff"] = kills["victim_placement"] - kills["killer_placement"]
winners = dict(kills[kills["killer_placement"] == 1][["killer_name", "match_id"]].groupby("match_id").size())
kills["winner_kills"] = kills["match_id"].map(winners, kills["match_id"])
kills.head()
def remove_outliers(kills, var):

    lower_bound = np.quantile(kills[var], 0.25) - (1.5 * (np.quantile(kills[var], 0.75) - np.quantile(kills[var], 0.25)))

    upper_bound = np.quantile(kills[var], 0.75) + (1.5 * (np.quantile(kills[var], 0.75) - np.quantile(kills[var], 0.25)))

    kills = kills[(kills[var] >= lower_bound) & (kills[var] <= upper_bound)]

    return kills



def distribution_cont(kills, var):

    if var == "kill_distance":

        kills = remove_outliers(kills, var)

    plt.hist(kills[var])

    plt.axvline(x = np.mean(kills[var]), c = "r")

    plt.text(0.7, 0.9, f"Mean: {round(np.mean(kills[var]), 2)}", transform = plt.gca().transAxes)

    plt.xlabel(f"{var}")

    plt.ylabel("Count")

    plt.title(f"Distribution of {var}")

    plt.show()

    

def basic_plot(kills, var):

    if var == "killed_by":

        grouped = kills.groupby(var).size().sort_values(ascending = False).head(10)

        plt.xticks(rotation = 90)

    else:

        grouped = kills.groupby(var).size().sort_values(ascending = False)

    plt.bar(grouped.index.values, grouped.values)

    plt.title(f"{var} Distribution")

    plt.show()
[distribution_cont(kills, i) for i in ["killer_placement", "victim_placement", "place_diff"]]
[distribution_cont(kills, i) for i in ["killer_position_x", "killer_position_y", "victim_position_x", "victim_position_y", "kill_distance"]]
[distribution_cont(kills, i) for i in ["time"]]
[distribution_cont(kills, i) for i in ["winner_kills"]]
basic_plot(kills, "map")
basic_plot(kills, "killed_by")
erangel = kills[kills["map"] == "ERANGEL"]
erangel_sample = erangel.sample(n = 5, random_state = 1)
img = plt.imread("../input/pubg-match-deaths/erangel.jpg")

plt.subplots(figsize = (12,12))

plt.imshow(img,aspect='auto', extent=[0, 800000, 0, 800000])

plt.scatter(erangel_sample["killer_position_x"], erangel_sample["killer_position_y"], c = "r", s = 100)

plt.scatter(erangel_sample["victim_position_x"], erangel_sample["victim_position_y"], c = "g", s = 100)

plt.legend(["Killer", "Victim"], loc = "upper right")

plt.show()
miramar = kills[kills["map"] == "MIRAMAR"]
miramar_sample = miramar.sample(n = 5, random_state = 1)
img = plt.imread("../input/pubg-match-deaths/miramar.jpg")

plt.subplots(figsize = (12,12))

plt.imshow(img, aspect='auto', extent=[0, 800000, 0, 800000])

plt.scatter(miramar_sample["killer_position_x"], miramar_sample["killer_position_y"], c = "r", s = 100)

plt.scatter(miramar_sample["victim_position_x"], miramar_sample["victim_position_y"], c = "g", s = 100)

plt.legend(["Killer", "Victim"], loc = "upper right")

plt.show()
def get_player_movement(kills, match_id, player_name):

    game = kills[kills["match_id"] == match_id]

    player = game[(game["killer_name"] == player_name) | (game["victim_name"] == player_name)].sort_values(by = "time")

    m = player.iloc[0]["map"]

    place = int(player.iloc[0]["killer_placement"])

    kills = len(list(player[player["killer_name"] == player_name]["killer_name"]))

    killer_name = player.iloc[-1]["killer_name"]

    killed_by = player.iloc[-1]["killed_by"]

    

    if m == "ERANGEL":

        img = plt.imread("../input/pubg-match-deaths/erangel.jpg")

    else:

        img = plt.imread("../input/pubg-match-deaths/miramar.jpg")

    plt.subplots(figsize = (12,12))

    plt.imshow(img, aspect='auto', extent=[0, 800000, 0, 800000])

    plt.scatter(player["killer_position_x"].iloc[0:-2], player["killer_position_y"].iloc[0:-2], c = "g", s = 100)

    plt.scatter(player["victim_position_x"].iloc[-1], player["victim_position_y"].iloc[-1], c = "r", s = 100)

    if place != 1:

        plt.title(f"{player_name}'s Game (Place: #{place}, Kills: {kills}, Killed by {killer_name} with {killed_by})")

        plt.legend(["Kill", "Point of Death"], loc = "upper right")

    else:

        plt.title(f"{player_name}'s Game (Place: #{place}, Kills: {kills})")

        plt.legend(["Kill", "Last Kill"], loc = "upper right")
rand_match = kills.iloc[r.randint(0, 11908315)]
get_player_movement(kills, rand_match["match_id"], rand_match["killer_name"])
rand_match2 = kills.iloc[r.randint(0, 11908315)]
get_player_movement(kills, rand_match2["match_id"], rand_match2["killer_name"])
rand_match3 = kills.iloc[r.randint(0, 11908315)]
get_player_movement(kills, rand_match3["match_id"], rand_match3["killer_name"])
distances = remove_outliers(kills, "kill_distance")
sns.violinplot(data = distances, y = "kill_distance")

plt.title("Kill Distance Distribution")

plt.show()
def prop_list(lst):

    weapons = list(lst)

    weapon_props = []

    distinct_weapons = list(set(weapons))

    for i in distinct_weapons:

        counter = 0

        for j in weapons:

            if i == j:

                counter += 1

        weapon_props.append(counter / len(weapons))

    return dict(zip(distinct_weapons, weapon_props))
distances["weapon_prop"] = distances["killed_by"].map(prop_list(distances["killed_by"]), distances["killed_by"])
def get_weapons_range(distances, perc_low, perc_high):

    lower_quantile = np.quantile(distances["kill_distance"], perc_low)

    upper_quantile = np.quantile(distances["kill_distance"], perc_high)

    distances2 = distances[(distances["kill_distance"] >= lower_quantile) & (distances["kill_distance"] <= upper_quantile)]

    

    weapon_counts = distances2.groupby("killed_by").size().sort_values(ascending = False).head(10)

    

    distances2["weapon_prop_filtered"] = distances2["killed_by"].map(prop_list(distances2["killed_by"]), distances2["killed_by"])

    distances2["conditional_prob"] = (distances2["weapon_prop_filtered"] - distances2["weapon_prop"])

    distinct_props = distances2[["killed_by", "conditional_prob"]].drop_duplicates().sort_values(by = "conditional_prob", ascending = False).head(10)

    

    plt.subplots(figsize = (10,6))

    plt.subplot(1,2,1)

    sns.barplot(weapon_counts.index.values, weapon_counts.values)

    plt.xticks(rotation = 90)

    plt.title(f"By Frequency ({perc_low} to {perc_high} Percentile)")

    plt.subplot(1,2,2)

    sns.barplot(distinct_props["killed_by"], distinct_props["conditional_prob"])

    plt.xticks(rotation = 90)

    plt.title(f"By Proportion Diff ({perc_low} to {perc_high} Percentile)")

    plt.xlabel(None)

    plt.ylabel(None)

    plt.show()    
get_weapons_range(distances, 0, 0.05)
get_weapons_range(distances, 0.05, 0.25)
get_weapons_range(distances, 0.25, 0.75)
get_weapons_range(distances, 0.75, 1)
kills.head()
sns.violinplot(data = kills, y = "time")

plt.title("Time Distribution")

plt.show()
def weapons_time(kills, min_time, max_time = max(list(kills["time"]))):

    

    time_range = kills[(kills["time"] >= min_time) & (kills["time"] <= max_time)]

    grouped_time = time_range.groupby("killed_by").size().sort_values(ascending = False)

    top_10 = grouped_time.head(10)

    deaths = sum(list(grouped_time.values))

    

    sns.barplot(top_10.index.values, top_10.values)

    plt.title(f"Top 10 Weapons ({min_time} to {max_time} sec.); {deaths} Total Deaths")

    plt.xticks(rotation = 90)

    plt.show()
weapons_time(kills, 0, 60)
weapons_time(kills, 0, 180)
weapons_time(kills, np.quantile(kills["time"], 0.25), np.quantile(kills["time"], 0.75))
weapons_time(kills, np.quantile(kills["time"], 0.9))
final_kills = kills[(kills["killer_placement"] == 1) & (kills["victim_placement"] == 2)]
grouped_final_kills = final_kills.groupby("killed_by").size().sort_values(ascending = False).head(10)
sns.barplot(grouped_final_kills.index.values, grouped_final_kills.values)

plt.xticks(rotation = 90)

plt.title("Top 10 Weapons For Final Kills")

plt.show()
inc_wins = kills
place = prop_list(inc_wins["killer_placement"])
inc_wins["kill_prop"] = inc_wins["killer_placement"].map(place, inc_wins["killer_placement"])
place_df = inc_wins[["killer_placement", "kill_prop"]].drop_duplicates().sort_values(by = "killer_placement", ascending = False)
plt.scatter(place_df["kill_prop"], place_df["killer_placement"])

plt.plot(place_df["kill_prop"], place_df["killer_placement"])

plt.xlabel("Proportion of Kills")

plt.ylabel("Placement (1-100)")

plt.title("Overall Proportion of Kills by Placement")

plt.show()
miramar_wins = inc_wins[inc_wins["map"] == "MIRAMAR"]

erangel_wins = inc_wins[inc_wins["map"] == "ERANGEL"]

place_m = prop_list(miramar_wins["killer_placement"])

place_e = prop_list(erangel_wins["killer_placement"])

miramar_wins["kill_prop"] = miramar_wins["killer_placement"].map(place_m, miramar_wins["killer_placement"])

erangel_wins["kill_prop"] = erangel_wins["killer_placement"].map(place_e, erangel_wins["killer_placement"])

place_df_m = miramar_wins[["killer_placement", "kill_prop"]].drop_duplicates().sort_values(by = "killer_placement", ascending = False)

place_df_e = erangel_wins[["killer_placement", "kill_prop"]].drop_duplicates().sort_values(by = "killer_placement", ascending = False)
plt.subplot(1,2,1)

plt.scatter(place_df_e["kill_prop"], place_df_e["killer_placement"])

plt.plot(place_df_e["kill_prop"], place_df_e["killer_placement"])

plt.xlabel("Proportion of Kills")

plt.title("Erangel")

plt.subplot(1,2,2)

plt.scatter(place_df_m["kill_prop"], place_df_m["killer_placement"])

plt.plot(place_df_m["kill_prop"], place_df_m["killer_placement"])

plt.xlabel("Proportion of Kills")

plt.title("Miramar")

plt.show()