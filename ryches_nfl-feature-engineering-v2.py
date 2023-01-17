import pandas as pd

train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")
first_play_id = train["PlayId"].unique()[0]
first_play = train[train["PlayId"]==first_play_id]
formation_cols = ["X", "Y", "S", "A", "Dis", "Orientation", "Dir", "OffenseFormation",

                  "OffensePersonnel", "DefendersInTheBox", "Position"]
first_play[formation_cols]
first_play.groupby("Team")[["X", "Y"]].mean()
xy_diff = first_play.groupby("Team")[["X", "Y"]].mean().diff().iloc[-1]
xy_diff
team_widths = first_play.groupby("Team")[["Y"]].std()

team_widths
diff_widths = first_play.groupby("Team")[["Y"]].std().diff().iloc[-1]

diff_widths
first_play.groupby("Team").agg(["max", "min", "std", "mean"])["Dis"]
def correct_height(row):

    split_height = row["PlayerHeight"].split("-")

    ft = split_height[0]

    inch = split_height[1]

    height = int(ft)*12 + int(inch)

    return height

first_play["CorrectedHeight"] = first_play.apply(correct_height, axis = 1)
first_play.groupby("Team").agg(["max", "min", "mean", "std"])[["CorrectedHeight", "PlayerWeight"]]
def correct_age(row):

    year_handoff = int(row["TimeHandoff"].split("-")[0])

    year_birth = int(row["PlayerBirthDate"].split("/")[-1])

    return year_handoff - year_birth

first_play["PlayerAge"] = first_play.apply(correct_age, axis = 1)
first_play.groupby("Team")["PlayerAge"].agg(["mean", "min", "std", "max"])
first_play[first_play["Position"] == "QB"]["PlayerAge"]
nflidvectors = train.groupby("NflIdRusher").mean()[["X", "Y", "S", "A", "Dis", "Orientation", "Dir", "Down", "Distance", "Quarter", "YardLine"]]
nflidvectors
first_play.merge(nflidvectors, left_on = "NflIdRusher", right_index = True)
player_df[comp_features]
from tqdm import tqdm

import numpy as np

closest_defender = []

comp_features = ["X", "Y", "CorrectedHeight", "PlayerWeight", "S", "A"]

for i in range(22):

    player_df = first_play.iloc[i]

    dist_df = (first_play[['X', 'Y']] - np.array(player_df[["X", "Y"]].values)).pow(2).sum(1).pow(0.5)

    first_play["player_dist"] = dist_df

    opponent_df = first_play[first_play["Team"] != player_df["Team"]]

    closest_df = opponent_df.loc[opponent_df["player_dist"].idxmin()]

    closest_defender.append(player_df[comp_features] - closest_df[comp_features])
pd.concat(closest_defender, axis = 1).T