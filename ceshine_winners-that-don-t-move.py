import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/train_V2.csv")
df["totalDistance"] = (
    df.rideDistance + df.walkDistance + df.swimDistance
)

df_tmp = df[(df.totalDistance == 0) & (df.winPlacePerc == 1)]
df_tmp.shape
problematic_matches = df_tmp.matchId.unique()
df_problems = df[df.matchId.isin(problematic_matches)]
df_matches = df_problems.groupby("matchId")[[
    "numGroups", "maxPlace", "matchType"]].first().join(
    df_problems.groupby("matchId")["maxPlace"].size().to_frame("player_count")
).join(
    df_problems.groupby("matchId")[["totalDistance"]].sum()
)
df_matches.describe()
df_matches
df_matches.numGroups.value_counts()
df_matches.groupby(["numGroups", "matchType"])["maxPlace"].count()

df_problems[
    (df_problems.numGroups == 2) & 
    (df_problems.matchType == "solo") & 
    (df_problems.winPlacePerc == 1)
].groupby(
    ["matchId", "groupId"]).size().value_counts()
df_problems[
    (df_problems.numGroups == 2) & 
    (df_problems.matchType == "solo") & 
    (df_problems.winPlacePerc < 1)
].groupby(
    ["matchId", "groupId"]).size().value_counts()