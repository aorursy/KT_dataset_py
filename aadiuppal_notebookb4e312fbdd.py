# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



matches = pd.read_csv('../input/matches.csv')

matches.head()

# Any results you write to the current directory are saved as output.
deliveries = pd.read_csv('../input/deliveries.csv')

deliveries.head()
team_score = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().unstack().reset_index()

team_score.columns = ['match_id', 'Team1_score', 'Team2_score', 'Team1_superover_score', 

                      'Team2_superover_score']

matches_agg = pd.merge(matches, team_score, left_on = 'id', right_on = 'match_id', how = 'outer')

team_extras = deliveries.groupby(['match_id', 'inning'])['extra_runs'].sum().unstack().reset_index()

team_extras.columns = ['match_id', 'Team1_extras', 'Team2_extras', 'Team1_superover_extras',

                       'Team2_superover_extras']

matches_agg = pd.merge(matches_agg, team_extras, on = 'match_id', how = 'outer')

#matches_agg.head()



batsman_grp = deliveries.groupby(["match_id", "inning", "batting_team", "batsman"])

batsmen = batsman_grp["batsman_runs"].sum().reset_index()

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

batsmen['SR'] = np.round(batsmen['batsman_runs'] / batsmen['balls_faced'] * 100, 2)

for col in ["batsman_runs", "4s", "6s", "balls_faced", "SR"]:

    batsmen[col] = batsmen[col].fillna(0)



dismissals = deliveries[ pd.notnull(deliveries["player_dismissed"])]

dismissals = dismissals[["match_id", "inning", "player_dismissed", "dismissal_kind", "fielder"]]

dismissals.rename(columns={"player_dismissed": "batsman"}, inplace=True)

batsmen = batsmen.merge(dismissals, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen = matches[['id','season']].merge(batsmen, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

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



bowlers_over = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler'])['over'].count().reset_index()

bowlers = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler']).sum().reset_index().drop('over', 1)

bowlers = bowlers_over.merge(bowlers, on=["match_id", "inning", "bowling_team", "bowler"], how = 'left')

bowlers['Econ'] = np.round(bowlers['runs'] / bowlers['over'] , 2)

bowlers = matches[['id','season']].merge(bowlers, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)



bowlers.head()
wins_home_away = matches_agg.groupby(['winner','city'])

plot = wins_home_away.plot(kind='bar')