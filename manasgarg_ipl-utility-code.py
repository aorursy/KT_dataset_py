import numpy as np

import pandas as pd



matches = pd.read_csv( "../input/matches.csv")

deliveries = pd.read_csv( "../input/deliveries.csv")
# Add the aggregates related to various runs



team_score_grp = deliveries.groupby( ["match_id", "inning"])



team_scores = team_score_grp["wide_runs", "bye_runs", "legbye_runs", 

                             "noball_runs", "penalty_runs", "extra_runs", 

                             "total_runs"].sum().unstack().reset_index()



team_scores.index = team_scores.match_id

team_scores = team_scores.fillna(0)



def merge_run_aggregates( matches_frame, runs_frame, run_category):

    run_category_frame = runs_frame[run_category].reset_index()

    run_category_frame.columns = ["match_id", "team1_" + run_category, 

                                  "team2_" + run_category, 

                                  "team1_superover_" + run_category, 

                                  "team2_superover_" + run_category]

    matches_frame = matches_frame.merge(run_category_frame, left_on="id",

                                        right_on="match_id", how="outer")

    del( matches_frame["match_id"])

    return matches_frame



for run_category in ["wide_runs", "bye_runs", "legbye_runs", "noball_runs",

                     "penalty_runs", "extra_runs", "total_runs"]:

    matches = merge_run_aggregates( matches, team_scores, run_category)
# Add the wicket aggregates at match/inning level



deliveries["is_wicket"] = pd.notnull(deliveries["player_dismissed"]).astype(int)



wicket_grp = deliveries.groupby(["match_id", "inning"])["is_wicket"].sum().unstack().reset_index()

wicket_grp = wicket_grp.fillna(0)

wicket_grp.columns = ["match_id", "team1_wickets", "team2_wickets", 

                      "team1_superover_wickets", "team2_superover_wickets"]

matches = matches.merge(wicket_grp, left_on="id", right_on="match_id", how="outer")

del( matches["match_id"])
# All our aggregates are now part of the matches dataframe

matches.columns
matches["type"] = "pre-qualifier"

for year in range(2008, 2017):

    final_match_index = matches[matches['season']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "qualifier-2")

    matches = matches.set_value(final_match_index-2, "type", "eliminator")

    matches = matches.set_value(final_match_index-3, "type", "qualifier-1")



matches.groupby(["type"])["id"].count()
batsman_grp = deliveries.groupby(["match_id", "inning", "batting_team", "batsman"])

batsmen = batsman_grp["batsman_runs"].sum().reset_index()



# Ignore the wide balls.

balls_faced = deliveries[deliveries["wide_runs"] == 0]

balls_faced = balls_faced.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

balls_faced.columns = ["match_id", "inning", "batsman", "balls_faced"]

batsmen = batsmen.merge(balls_faced, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")
fours = deliveries[ deliveries["batsman_runs"] == 4]

sixes = deliveries[ deliveries["batsman_runs"] == 6]



fours_per_batsman = fours.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

sixes_per_batsman = sixes.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()



fours_per_batsman.columns = ["match_id", "inning", "batsman", "4s"]

sixes_per_batsman.columns = ["match_id", "inning", "batsman", "6s"]



batsmen = batsmen.merge(fours_per_batsman, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen = batsmen.merge(sixes_per_batsman, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")



for col in ["batsman_runs", "4s", "6s", "balls_faced"]:

    batsmen[col] = batsmen[col].fillna(0)
dismissals = deliveries[ pd.notnull(deliveries["player_dismissed"])]

dismissals = dismissals[["match_id", "inning", "player_dismissed", "dismissal_kind", "fielder"]]

dismissals.rename(columns={"player_dismissed": "batsman"}, inplace=True)

batsmen = batsmen.merge(dismissals, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")
batsmen.head()
bowler_grp = deliveries.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])

bowlers = bowler_grp["total_runs", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"].sum().reset_index()



bowlers["runs"] = bowlers["total_runs"] - (bowlers["bye_runs"] + bowlers["legbye_runs"])

bowlers["extras"] = bowlers["wide_runs"] + bowlers["noball_runs"]



del( bowlers["bye_runs"])

del( bowlers["legbye_runs"])

del( bowlers["total_runs"])
dismissal_kinds_for_bowler = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]

dismissals = deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds_for_bowler)]

dismissals = dismissals.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])["dismissal_kind"].count().reset_index()

dismissals.rename(columns={"dismissal_kind": "wickets"}, inplace=True)



bowlers = bowlers.merge(dismissals, left_on=["match_id", "inning", "bowling_team", "bowler", "over"], 

                        right_on=["match_id", "inning", "bowling_team", "bowler", "over"], how="left")

bowlers["wickets"] = bowlers["wickets"].fillna(0)
bowlers.head()
tmpgrp = bowlers.groupby(["match_id", "inning", "bowling_team", "bowler"])

bowlers_per_match = tmpgrp["runs", "extras", "wide_runs", "noball_runs", "wickets"].sum().reset_index()
bowlers_per_match.head()