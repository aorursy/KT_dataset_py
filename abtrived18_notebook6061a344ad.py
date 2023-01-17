import pandas as pd
pd.set_option('max_rows',200)
pd.set_option('max_columns',100)
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
deliveries = pd.read_csv('../input/ipldata/deliveries.csv')
matches = pd.read_csv('../input/ipldata/matches.csv')

city = matches.groupby([matches.city]).city.count().sort_values(ascending=False)
plt.figure(figsize=(15,5))
city.plot(kind="bar")

#matches['season'].unique()
matches.city.fillna('unknown',inplace=True)
matches.winner.fillna('draw',inplace=True)
matches.player_of_match.fillna('draw',inplace=True)
matches['season'].unique()
winner = matches.groupby(matches.winner).winner.count().sort_values(ascending=False)
plt.figure(figsize=(15,5))
winner.plot(kind='bar')
s_man_of_match = matches.groupby(matches.player_of_match).player_of_match.count().sort_values(ascending=False).head(15)
df_man_of_match =s_man_of_match.to_frame()
df_man_of_match
#deliveries.groupby(deliveries.batsman).batsman_runs.sum().sort_values(ascending=False).head(15)
df_strike_rate = deliveries.groupby(['batsman']).agg({'ball':'count','batsman_runs':'mean'}).sort_values(by='batsman_runs',ascending=False)
df_strike_rate.rename(columns ={'batsman_runs' : 'strike rate'}, inplace=True)
df_runs_per_match = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})
df_total_runs = df_runs_per_match.groupby(['batsman']).agg({'sum' ,'mean','count'})
df_total_runs.rename(columns ={'sum' : 'batsman run','count' : 'match count','mean' :'average score'}, inplace=True)
df_total_runs.columns = df_total_runs.columns.droplevel()
df_sixes = deliveries[['batsman','batsman_runs']][deliveries.batsman_runs==6].groupby(['batsman']).agg({'batsman_runs':'count'})
df_four = deliveries[['batsman','batsman_runs']][deliveries.batsman_runs==4].groupby(['batsman']).agg({'batsman_runs':'count'})
df_batsman_stat = pd.merge(pd.merge(pd.merge(df_strike_rate,df_total_runs, left_index=True, right_index=True),df_sixes, left_index=True, right_index=True),df_four, left_index=True, right_index=True)
#print(df_batsman_stat)
df_batsman_stat.rename(columns = {'ball' : 'Ball', 'strike rate':'Strike Rate','batsman run' : 'Batsman Run','match count' : 'Match Count','average score' : 'Average score' ,'batsman_runs_x' :'Six','batsman_runs_y':'Four'},inplace=True)
df_batsman_stat['Strike Rate'] = df_batsman_stat['Strike Rate']*100
df_batsman_stat.sort_values(by='Batsman Run',ascending=False).head(25)

condition = (deliveries.dismissal_kind.notnull()) &(deliveries.dismissal_kind != 'run out')&(deliveries.dismissal_kind != 'retired hurt')
condition_fielding = (deliveries.dismissal_kind == 'caught') | (deliveries.dismissal_kind == 'run out')
df_bowlers = deliveries.loc[condition,:].groupby(deliveries.bowler).dismissal_kind.count().sort_values(ascending=False)
df_runs_match = deliveries.groupby(['bowler','match_id']).agg({'total_runs':'sum','ball':'count',})
#df_runs_matchs = df_runs_match.columns.droplevel()
#df_bowlers.head(15)
df_runs_match.total_runs = df_runs_match.total_runs
df_runs_match['run_Rate'] = df_runs_match.total_runs/df_runs_match.ball*6
#df_runRate = df_runRate.groupby(['bowler']).agg({'run_Rate':'sum'})

#df_runs_match.sort_values(by='total_runs',ascending=True)
df_runRate = df_runs_match.run_Rate.groupby(['bowler']).agg({'mean'})
df_bowlers = pd.merge(df_bowlers.to_frame(),df_runRate , how='inner', left_index=True, right_index=True)
df_bowlers.rename({'mean':'Run Rate'}, axis=1, inplace=True)
df_bowlers.head(20)
s_fielding = deliveries.loc[condition_fielding,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)
df_fielding= s_fielding.to_frame()
df_fielding.columns = ['fielding']
df_fielding.head(15)
#df_points = g5[['ball','Strike Rate','match count','average score','batsman run','six','Man of Match']]
df_points = df_batsman_stat[['Ball','Strike Rate','Average score','Batsman Run','Six']]
#print(df_points.head(10))
df_points['Six pts'] = df_points['Six']*2
df_points = pd.merge(pd.merge(df_points.merge(df_bowlers, left_index=True, right_index=True), df_fielding, left_index=True, right_index=True, how='left'), df_man_of_match, left_index=True, right_index=True, how='outer')
df_points.fillna(value=0, inplace=True)
df_points['dismissal_kind'] = df_points['dismissal_kind'] *20
df_points['fielding'] = df_points['fielding'] *10
#print(df_points.head(10))
df_points['Man of Match pts'] = df_points['player_of_match']*25
df_points['Total Point'] = df_points['Batsman Run']+ df_points['Six']+df_points['dismissal_kind']+df_points['Man of Match pts']+df_points['fielding']
df_points = df_points.drop('player_of_match',1)
df_points = df_points.drop('Six',1)
df_points.rename({'dismissal_kind':'bowling pts'}, axis=1, inplace=True)
df_points.rename({'fielding':'fielding pts'}, axis=1, inplace=True)
df_points.rename({'batsman run':'batsman run pts'}, axis=1, inplace=True)
#df_points['Points per match'] =df_points['Total Point']/df_points['match count']
df_points.sort_values(by='Total Point',ascending=False,inplace=True)
df_points.head(25)
combined_df = pd.merge(deliveries, matches, how='outer', left_on='match_id', right_on='id')
venue = 'Wankhede Stadium'
stadium_df = combined_df[:][combined_df.venue == venue]
rating_batsman  = stadium_df.groupby(stadium_df.batsman).batsman_runs.sum().sort_values(ascending=False).head(10)
#rating_batsman.plot(kind='bar')
print(rating_batsman)
condition = (stadium_df.dismissal_kind.notnull()) &(stadium_df.dismissal_kind != 'run out')&(stadium_df.dismissal_kind != 'retired hurt')&(stadium_df.dismissal_kind != 'notout')

rating_bowler  = stadium_df.loc[condition,:].groupby(stadium_df.bowler).dismissal_kind.count().sort_values(ascending=False).head(10)
rating_bowler.plot(kind='bar')