import pandas as pd

import numpy as np
t20_data = pd.read_csv("../input/T20_matches_ball_by_ball_data.csv", 

                       parse_dates=["date"], low_memory=False)
t20_data.isnull().sum()
t20_data["runs_plus_extras"] = t20_data["Run_Scored"] + t20_data["Extras"]

t20_data["total_runs"] = t20_data.groupby(["Match_Id", "Batting_Team"])["runs_plus_extras"].cumsum()
t20_data["wicket_fall"] = t20_data["Dismissed"].isnull().map({True: 0, False: 1})

t20_data["wickets"] = t20_data.groupby(["Match_Id", "Batting_Team"])["wicket_fall"].cumsum()
runs_scored = t20_data.groupby(["Match_Id", "Batting_Team", "Striker"], as_index=False)["Run_Scored"].sum()

balls_faced = t20_data.groupby(["Match_Id", "Batting_Team", "Striker"], as_index=False)["Run_Scored"].count()

balls_faced.columns = ["Match_Id", "Batting_Team", "Striker", "Balls"]

batting_scoreboard = pd.merge(runs_scored, balls_faced, 

                              on=["Match_Id", "Batting_Team", "Striker"], how="left")
t20_dismissal = t20_data[["Match_Id", "Batting_Team", "Striker", "Dismissal"]]

t20_dismissal["concat_key"] = t20_dismissal["Match_Id"].map(str) + ":" + t20_dismissal["Striker"]

t20_dismissal = t20_dismissal.drop_duplicates(subset=["concat_key"], keep="last")

t20_dismissal = t20_dismissal.drop(labels="concat_key", axis = 1)

t20_dismissal = t20_dismissal.sort_values(["Match_Id", "Batting_Team"])

t20_dismissal.Dismissal.fillna("not out", inplace=True)
batting_scoreboard = pd.merge(batting_scoreboard, t20_dismissal, 

                              on=["Match_Id", "Batting_Team", "Striker"], how="left")

batting_scoreboard.head()
batsman_statistics = pd.DataFrame({"Batsman": batting_scoreboard.Striker.unique()})
Innings = pd.DataFrame(batting_scoreboard.Striker.value_counts())

Innings.reset_index(inplace=True)

Innings.columns = ["Batsman", "Innings"]
Not_out = batting_scoreboard.Dismissal == "not out"

batting_scoreboard["Not_out"] = Not_out.map({True: 1, False: 0})

Not_out = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Not_out"].sum())

Not_out.reset_index(inplace=True)

Not_out.columns = ["Batsman", "Not_out"]


Balls = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Balls"].sum())

Balls.reset_index(inplace=True)

Balls.columns = ["Batsman", "Balls"]
Run_Scored = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Run_Scored"].sum())

Run_Scored.reset_index(inplace=True)

Run_Scored.columns = ["Batsman", "Run_Scored"]
Highest_Score = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Run_Scored"].max())

Highest_Score.reset_index(inplace=True)

Highest_Score.columns = ["Batsman", "Highest_Score"]
Centuries = pd.DataFrame(batting_scoreboard.loc[batting_scoreboard.Run_Scored >= 100,].groupby(["Striker"])["Run_Scored"].count())

Centuries.reset_index(inplace=True)

Centuries.columns = ["Batsman", "Centuries"]
Half_Centuries = pd.DataFrame(batting_scoreboard.loc[(batting_scoreboard.Run_Scored >= 50) & 

                                                     (batting_scoreboard.Run_Scored < 100),].groupby(["Striker"])["Run_Scored"].count())

Half_Centuries.reset_index(inplace=True)

Half_Centuries.columns = ["Batsman", "Half_Centuries"]
batsman_statistics = pd.merge(batsman_statistics, Innings, on=["Batsman"], how="left")

batsman_statistics = pd.merge(batsman_statistics, Not_out, on=["Batsman"], how="left")

batsman_statistics = pd.merge(batsman_statistics, Balls, on=["Batsman"], how="left")

batsman_statistics = pd.merge(batsman_statistics, Run_Scored, on=["Batsman"], how="left")

batsman_statistics = pd.merge(batsman_statistics, Highest_Score, on=["Batsman"], how="left")



batsman_statistics = pd.merge(batsman_statistics, Centuries, on=["Batsman"], how="left")

batsman_statistics.Centuries.fillna(0, inplace=True)

batsman_statistics.Centuries = batsman_statistics.Centuries.astype("int")



batsman_statistics = pd.merge(batsman_statistics, Half_Centuries, on=["Batsman"], how="left")

batsman_statistics.Half_Centuries.fillna(0, inplace=True)

batsman_statistics.Half_Centuries = batsman_statistics.Half_Centuries.astype("int")
batsman_statistics["Batting_Average"] = batsman_statistics.Run_Scored / (batsman_statistics.Innings - batsman_statistics.Not_out)

batsman_statistics.loc[batsman_statistics["Batting_Average"] == np.inf, "Batting_Average"] = 0

batsman_statistics.loc[batsman_statistics["Batting_Average"].isnull(), "Batting_Average"] = 0
batsman_statistics["Strike_Rate"] = (batsman_statistics.Run_Scored * 100) / batsman_statistics.Balls
batsman_statistics = batsman_statistics.round({"Batting_Average": 2, "Strike_Rate": 2})
batsman_statistics.head()