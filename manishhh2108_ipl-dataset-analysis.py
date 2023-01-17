import pandas as pd

pd.set_option('max_rows',200)

pd.set_option('max_columns',100)

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
deliveries = pd.read_csv('../input/deliveries.csv')

matches = pd.read_csv('../input/matches.csv')
matches.city.fillna('unknown',inplace=True)

matches.winner.fillna('draw',inplace=True)

matches.player_of_match.fillna('draw',inplace=True)
matches.count()
matches.groupby(matches.player_of_match).player_of_match.count().sort_values(ascending=False).head(5)
teams = matches.team1.append(matches.team2)

teams_df = pd.DataFrame(teams)

teams_df.columns = ['team']

count_match_series = teams_df.groupby(teams_df.team).team.count()

count_match_df = pd.DataFrame(count_match_series)

count_match_df.columns = ['countplayed']

count_match_df.reset_index(inplace=True)

count_match_df.sort_values('team')

team_sorted = count_match_df.team

played_sorted = count_match_df.countplayed
count_match_df
winner_series = matches.groupby(matches.winner).winner.count()

winner_df = pd.DataFrame(winner_series)

winner_df.columns = ['countwin']

winner_df.reset_index(inplace=True)

win_sorted = winner_df.countwin
winner_df
total_losses = played_sorted - win_sorted

loss_df = pd.DataFrame(total_losses)

team_df = pd.DataFrame(team_sorted)

loss_df = pd.concat((loss_df,team_df),axis=1)

loss_df.columns = ['loss_count','team']

loss_df.sort_values('loss_count',ascending=False)
matches.groupby(matches.toss_winner).toss_winner.count()
choice_series = matches.groupby(matches.toss_decision).toss_decision.count()

choice_series.plot(kind='bar')
condition = matches.toss_decision =='bat'

df1 = matches.loc[condition,:]

df2 = matches.loc[~condition,:]



percentage_when_batting_first_won  = (df1.toss_winner == df1.winner).mean()

percentage_when_fielding_first_won = (df2.toss_winner == df2.winner).mean()

df = pd.DataFrame([percentage_when_batting_first_won,percentage_when_fielding_first_won])

df.plot(kind='bar')
deliveries.dismissal_kind.fillna('notout',inplace=True)

deliveries.player_dismissed.fillna('notout',inplace=True)

deliveries.fielder.fillna('notout',inplace=True)
deliveries.count()


deliveries.dismissal_kind.unique()
condition = (deliveries.dismissal_kind != 'run out')&(deliveries.dismissal_kind != 'retired hurt')&(deliveries.dismissal_kind != 'notout')

df = deliveries.loc[condition,:]

df.groupby([df.over]).over.count().plot(kind='pie')
#deliveries.groupby(deliveries.bowling_team).total_runs.sum()
out_series = deliveries.groupby(deliveries.dismissal_kind).dismissal_kind.count()

out_df = pd.DataFrame(out_series).sort_values('dismissal_kind')

out_df
#deliveries.groupby(deliveries.bowler).total_runs.mean().sort_values()
condition = (deliveries.dismissal_kind != 'run out')&(deliveries.dismissal_kind != 'retired hurt')&(deliveries.dismissal_kind != 'notout')

df = deliveries.loc[condition,:]

#df.groupby(df.bowler).over.count().sort_values(ascending=False)
condition = (deliveries.over > 15) &(deliveries.dismissal_kind != 'run out')&(deliveries.dismissal_kind != 'retired hurt')&(deliveries.dismissal_kind != 'notout')

df = deliveries.loc[condition,:]

#df.groupby(df.bowler).over.count().sort_values(ascending=False)
#Including only those bowlers that have bowled atleast 15 death overs

del_series = deliveries.groupby(deliveries.bowler).total_runs.count()

del_df = pd.DataFrame(del_series).reset_index()

del_df.columns = ['bowler','delivery']

bowlers = del_df.loc[del_df.delivery >=90,:].bowler


condition = (deliveries.over > 15) & (deliveries.bowler.isin(bowlers))

df = deliveries.loc[condition,:]

df.groupby(df.bowler).total_runs.mean().sort_values().head(10)
#deliveries.groupby(deliveries.batting_team).total_runs.sum()
deliveries.groupby([deliveries.over]).total_runs.sum().plot(kind='pie')
#deliveries.groupby(deliveries.batsman).batsman_runs.sum().sort_values(ascending=False)
df = deliveries.loc[deliveries.inning == 2 ,:]

#df.groupby(deliveries.batsman).batsman_runs.sum().sort_values(ascending=False)
df = deliveries.loc[deliveries.batsman == 'V Kohli',:]

#df.groupby(df.bowling_team).batsman_runs.sum().sort_values(ascending=False)
df = deliveries.loc[deliveries.batsman == 'V Kohli',:]

#df.groupby(df.bowler).batsman_runs.sum().sort_values(ascending=False)



#Interesting find --Among top seven, six of them are indians----
runs_series = deliveries.groupby([deliveries.match_id,deliveries.batsman]).total_runs.sum()

runs_df = pd.DataFrame(runs_series)

runs_df = runs_df.reset_index()

condition = runs_df.total_runs >= 100

player_100df = runs_df.loc[condition,:]

player_100df.groupby(player_100df.batsman).total_runs.count().sort_values(ascending=False).head(5)
condition = deliveries.dismissal_kind == 'run out'

df = deliveries.loc[condition,:]

#df.groupby(df.fielder).batsman.count().sort_values(ascending=False)
condition = (deliveries.dismissal_kind == 'caught and bowled') |(deliveries.dismissal_kind == 'caught')

df = deliveries.loc[condition,:]

#df.groupby(df.fielder).batsman.count().sort_values(ascending=False)
condition = (deliveries.dismissal_kind == 'stumped')

df = deliveries.loc[condition,:]

#df.groupby(df.fielder).batsman.count().sort_values(ascending=False)
combined = deliveries.set_index('match_id').join(matches.set_index('id'),how='inner')
condition = combined.fielder != 'notout'

df = combined.loc[condition,:]

fielders_df = df.groupby([df.season,df.fielder]).non_striker.count()

fielders_df = pd.DataFrame(fielders_df)

fielders_df = fielders_df.reset_index()
condition = fielders_df.season == 2013

fielders_df.loc[condition,:].sort_values('non_striker',ascending=False).head(2)
runs_series = combined.groupby([combined.season,combined.batsman]).total_runs.sum()

runs_df = pd.DataFrame(runs_series)

runs_df.reset_index(inplace=True)
condition = runs_df.season == 2016

runs_df.loc[condition,:].sort_values('total_runs',ascending=False).head(5)
condition = (combined.dismissal_kind != 'run out')&(combined.dismissal_kind != 'retired hurt')&(combined.dismissal_kind != 'notout')

df = combined.loc[condition,:]

wicket_series = df.groupby([df.season,df.bowler]).non_striker.count()

wicket_df = pd.DataFrame(wicket_series)

wicket_df.reset_index(inplace=True)
condition = wicket_df.season == 2016

wicket_df.loc[condition,:].sort_values('non_striker',ascending=False).head(5)
condition = (combined.season==2016)&(combined.dismissal_kind != 'run out')&(combined.dismissal_kind != 'retired hurt')&(combined.dismissal_kind != 'notout')

df = combined.loc[condition,:]

wicket_series = df.groupby([df.index,df.bowler]).non_striker.count()

wicket_df = pd.DataFrame(wicket_series)

wicket_df.reset_index(inplace=True)
wicket_df.columns= ['match_id','bowler','wicket']

condition = wicket_df.wicket >= 4

wicket_df = wicket_df.loc[condition,:].sort_values('wicket',ascending=False)

wicket_df
df = pd.DataFrame(wicket_df.groupby(wicket_df.bowler).wicket.count().sort_values(ascending=False))

df.columns = ['4 Wickets']

df.reset_index()
#Filtering bowlers according to season

condition = combined.season == 2015

df = combined.loc[condition,:]
#Filtering bowlers that bowled minimum 60 balls

bowler_series = df.groupby([df.bowler]).total_runs.count()

bowler_df = pd.DataFrame(bowler_series)

bowler_df.columns = ['balls']

bowler_df = bowler_df.loc[bowler_df.balls >= 60,:]
condition = df.bowler.isin(bowler_df.index)

df2 = df.loc[condition,:]

df3 = pd.DataFrame(df2.groupby([df2.bowler]).total_runs.mean().sort_values().head(10))
df3.columns = ['Economy']

df3.reset_index()