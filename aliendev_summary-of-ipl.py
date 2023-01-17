# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

np.set_printoptions(precision=2)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
matches = pd.read_csv("../input/matches.csv")

deliveries = pd.read_csv("../input/deliveries.csv")
matches.head(1)
deliveries.head(1)
deliveries[deliveries.noball_runs >= 4]
matches.drop("umpire3", axis=1, inplace=True)
matches.head(1)
deliveries.fillna(value=0, inplace=True)
deliveries.columns
matches.columns
print("Total matches played between 2008-16 :",matches.shape[0])
print("Total umpires in IPL 2008-16 :",matches['umpire1'].nunique())
print("Total run scored b/w 2008-16: ",deliveries['total_runs'].sum())
print("Total venues used in IPL 2008-16:",matches['venue'].nunique())
print("Total host city IPL 2008-16:",matches['city'].nunique())
print("Total extra runs in IPL 2008-16:",deliveries['extra_runs'].agg(sum))
print("Total noball runs in IPL 2008-16:",deliveries['noball_runs'].agg(sum))
print("Total wide runs in IPL 2008-16:",deliveries['wide_runs'].agg(sum))
print("Top 10 venues where IPL matches host most of times")

matches['venue'].value_counts().sort_values(ascending=False)[:10]
print("Top 10 Team name who won toss most of times")

toss_winner_index = matches['toss_winner'].value_counts().index

toss_win_count = matches['toss_winner'].value_counts().tolist()

max_toss_winners = pd.DataFrame({'team_name':toss_winner_index, 'toss_win_count':toss_win_count})

max_toss_winners[:10]
print("%age of field/bat selection after toss winning")

(matches['toss_decision'].value_counts() / 577) * 100
temp_res = matches['venue'].value_counts()

pd.DataFrame({'count':temp_res.tolist(), 'venue':temp_res.index})[:10]
print("Top 10 Cities where IPL matches host most of times")

matches['city'].value_counts().sort_values(ascending=False)[:10]
print("Top 10 Team who won more matches")

temp_res = matches['winner'].value_counts()

most_winning_team = pd.DataFrame({'team_name':temp_res.index, 'winning_count':temp_res.tolist()})

most_winning_team[:10]
print("Teams name and total matches playes by them")

total_teams = pd.concat([matches['team1'],matches['team2']]).value_counts()

total_matches_played_by_team = pd.DataFrame({'team_name':total_teams.index, 'total_matches':total_teams.tolist()})

total_matches_played_by_team
print("Summary of team performance")

temp_res = total_matches_played_by_team.merge(most_winning_team, left_on="team_name", right_on="team_name", how='outer')

temp_res['winnig_percentage'] = (np.array(temp_res['winning_count'].tolist()) / np.array(temp_res['total_matches'].tolist())) * 100

temp_res = temp_res.merge(max_toss_winners, left_on="team_name", right_on="team_name", how='outer')

temp_res['toss_win_percetage'] = (np.array(temp_res['toss_win_count'].tolist()) / np.array(temp_res['total_matches'].tolist())) * 100

temp_res
print("Top 10 player of match")

matches['player_of_match'].value_counts().sort_values(ascending=False)[:10]
print("Matches played in each season:")

matches['season'].value_counts().sort_values(ascending=False)
print("Team 10 who w0n by wicket ")

temp_res = matches.groupby('team2')['win_by_wickets'].agg(max).sort_values(ascending=False)

win_by_wkt = pd.DataFrame({'team_name':temp_res.index, 'win_by_wickets':temp_res.tolist()})

win_by_wkt[:10]
print("Team 10 who won by runs")

temp_res = matches.groupby('team1')['win_by_runs'].agg(max).sort_values(ascending=False)

win_by_run = pd.DataFrame({'team_name':temp_res.index, 'win_by_runs':temp_res.tolist()})

win_by_run[:10]
print("Top 10 bowler who gave more runs:")

Bowler_run = deliveries.groupby('bowler')["total_runs"].agg(sum).reset_index().sort_values(by="total_runs",ascending=False)

Top_10_expen_bowler = Bowler_run[:10]

Top_10_expen_bowler.reset_index(drop=True)
print("Top dismissal_kind types")

index = deliveries['dismissal_kind'].value_counts().index.tolist()

count = deliveries['dismissal_kind'].value_counts().tolist()

pd.DataFrame({'dismissal_kind': index[1:], 'total_num':count[1:]})
print("Top 10 bowler who gave more extra runs")

total_extra = deliveries.groupby('bowler')['extra_runs'].agg(sum).reset_index().sort_values(by='extra_runs', ascending=False)

total_extra.reset_index(drop=True)[:10]
print("Top 10 Players who score more runs with there strike rates:")

balls=deliveries.groupby(['batsman'])['ball'].count().reset_index()

runs=deliveries.groupby(['batsman'])['batsman_runs'].sum().reset_index()

balls=balls.merge(runs,left_on='batsman',right_on='batsman',how='outer')

strike_rate = (balls['batsman_runs']/balls['ball']) * 100

balls['strike_rate'] = strike_rate;

balls.sort_values('batsman_runs', ascending=False).reset_index(drop=True)[:10]
print("Top 3 bowler who dismissed 'V Kholi' most of times")

deliveries['bowler'][deliveries['player_dismissed'] == 'V Kohli'].value_counts().sort_values(ascending=False)[:3]
print("Top 3 bowler who dismissed 'V Kholi' most of times")

deliveries['bowler'][deliveries['batsman'] == 'V Kohli'].value_counts().sort_values(ascending=False)[:3]
deliveries.groupby(['batting_team']).total_runs.agg(['sum','mean','max', 'min'])
print("Top 10 fielder who catch most")

f_c = deliveries['fielder'][deliveries['dismissal_kind'] == 'caught'].value_counts()[:10]

print(f_c)

f_c.plot(kind='bar', title='Player and their catch count')
print("Top 10 fielder who did most runout")

f_R = deliveries['fielder'][deliveries['dismissal_kind'] == 'run out'].value_counts()[:10]

print(f_R)

f_R.plot(kind='bar')
print("Batsman who score more than 4 run at noballs")

f_n = deliveries['batsman'][deliveries['noball_runs'] >= 4].value_counts()

print(f_n)

f_n.plot(kind='bar', title='No. of time score >= 4 at no balls')
deliveries