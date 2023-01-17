# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
# read the input files and look at the top few lines #

data_path = "../input/"

match_df = pd.read_csv(data_path+"matches.csv")

deliveries_df = pd.read_csv(data_path+"deliveries.csv")
match_df.head()
deliveries_df.head()
team_name_map = {'Mumbai Indians':'MI',                

'Royal Challengers Bangalore':'RCB' ,   

'Kings XI Punjab':'KXIP'    ,            

'Chennai Super Kings':'CSK' ,           

'Delhi Daredevils' :'DD'   ,           

'Kolkata Knight Riders'   :'KKR' ,      

'Rajasthan Royals'    :'RR'    ,      

'Deccan Chargers'    :'DC'   ,          

'Sunrisers Hyderabad' :'SRH' ,          

'Pune Warriors'       :'PWI' ,           

'Gujarat Lions'          :'GL',        

'Kochi Tuskers Kerala'      :'KT' ,   

'Rising Pune Supergiants':'RPS'}

deliveries_df['batting_team'] = deliveries_df['batting_team'].apply(lambda x: team_name_map[x])

deliveries_df['bowling_team'] = deliveries_df['bowling_team'].apply(lambda x: team_name_map[x])
deliveries_df['is_boundary'] = np.where(deliveries_df['batsman_runs'].isin([4,6]),1,0)

deliveries_df['is_wicket'] = np.where(deliveries_df['player_dismissed'].isnull(),0,1)

deliveries_df['is_bowler_wicket'] = np.where((deliveries_df['player_dismissed'].notnull())&(deliveries_df['dismissal_kind']!='run out'),1,0)

deliveries_df['phase'] = np.where(deliveries_df['over']<=6,'powerplay',np.where(deliveries_df['over']>15,'slog/death','middle'))
deliveries_df.columns
deliveries_df.head()
batting_grouped = deliveries_df.groupby(['batting_team','match_id','phase']).agg({'total_runs':sum,'ball':'count','is_wicket':sum,'is_boundary':sum,'over':pd.Series.nunique}).reset_index()
batting_grouped.head()
batting_grouped.columns = ['team','match','phase','runs_scored','balls_faced','wickets_lost','boundaries_scored','overs_faced']
batting_grouped['run_rate'] = batting_grouped['runs_scored']*1.0/batting_grouped['overs_faced']

batting_grouped['strike_rate'] = batting_grouped['runs_scored']*100.0/batting_grouped['balls_faced']
overall_run_rate = deliveries_df.groupby(['batting_team','match_id']).agg({'total_runs':sum,'over':pd.Series.nunique}).reset_index()
overall_run_rate.columns = ['team','match','runs_scored','overs_faced']

overall_run_rate.head()
overall_run_rate['match_run_rate'] = overall_run_rate['runs_scored']*1.0/overall_run_rate['overs_faced']

overall_avg_run_rate = overall_run_rate.groupby('team').agg({'match_run_rate':'mean','match':'count'}).reset_index()
agg = batting_grouped.groupby(['team','phase']).agg({'run_rate':'mean'}).reset_index()

table = agg.pivot(index='team', columns='phase', values='run_rate').reset_index()

table = table.merge(overall_avg_run_rate,on='team',how='left')
table = table[['team', 'powerplay','middle', 'slog/death', 'match_run_rate','match']]
table
bowling_grouped = deliveries_df.groupby(['bowling_team','match_id','phase']).agg({'total_runs':sum,'ball':'count','is_wicket':sum,'is_boundary':sum,'over':pd.Series.nunique}).reset_index()

bowling_grouped.columns = ['team','match','phase','runs_conceded','balls_bowled','wickets_taken','boundaries_conceded','overs_bowled']

bowling_grouped['econ_rate'] = bowling_grouped['runs_conceded']*1.0/bowling_grouped['overs_bowled']



overall_econ_rate = deliveries_df.groupby(['bowling_team','match_id']).agg({'total_runs':sum,'over':pd.Series.nunique}).reset_index()

overall_econ_rate.columns = ['team','match','runs_conceded','overs_bowled']

overall_econ_rate['match_econ_rate'] = overall_econ_rate['runs_conceded']*1.0/overall_econ_rate['overs_bowled']

overall_avg_econ_rate = overall_econ_rate.groupby('team').agg({'match_econ_rate':'mean','match':'count'}).reset_index()
agg = bowling_grouped.groupby(['team','phase']).agg({'econ_rate':'mean'}).reset_index()

table = agg.pivot(index='team', columns='phase', values='econ_rate').reset_index()

table = table.merge(overall_avg_econ_rate,on='team',how='left')

table = table[['team', 'powerplay','middle', 'slog/death', 'match_econ_rate','match']]

table
deliveries_df.columns
powerplay_df = deliveries_df[deliveries_df['phase']=='powerplay']
batsmen_powerplay_grouped = powerplay_df.groupby(['batsman','match_id']).agg({'batsman_runs':'sum','ball':'count'}).reset_index().rename(columns={'batsman_runs':'runs_scored','ball':'balls_faced'})

batsmen_powerplay_grouped['strikerate'] = batsmen_powerplay_grouped['runs_scored']*100.0/batsmen_powerplay_grouped['balls_faced']
agg = batsmen_powerplay_grouped.groupby('batsman').agg({'strikerate':'mean','balls_faced':'sum'}).reset_index().rename(columns={'strikerate':'avg_sr','balls_faced':'total_balls_faced'})

agg = agg[agg['total_balls_faced']>=300]

agg = agg.sort_values('avg_sr',ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
g = sns.barplot(x='batsman',y='avg_sr',data=agg.iloc[0:10,]);

fig = g.get_figure()

fig.set_size_inches(12,10)

fig.suptitle("Average Strikerate Top 10 Batsmen in Powerplay(played more than 300 balls)")

fig.tight_layout()

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 
bowler_powerplay_grouped = powerplay_df.groupby(['bowler','match_id']).agg({'total_runs':'sum','over':pd.Series.nunique}).reset_index().rename(columns={'total_runs':'runs_conceded','over':'overs_bowled'})

bowler_powerplay_grouped['economyrate'] = bowler_powerplay_grouped['runs_conceded']*1.0/bowler_powerplay_grouped['overs_bowled']
agg = bowler_powerplay_grouped.groupby('bowler').agg({'economyrate':'mean','overs_bowled':'sum'}).reset_index().rename(columns={'economyrate':'avg_er','overs_bowled':'num_overs_bowled'})

agg = agg[agg['num_overs_bowled']>=50]

agg = agg.sort_values('avg_er',ascending=True)
g = sns.barplot(x='bowler',y='avg_er',data=agg.iloc[0:10,]);

fig = g.get_figure()

fig.set_size_inches(12,10)

fig.suptitle("Top 10 Bowlers in Powerplay(bowled more than 50 overs)")

fig.tight_layout()

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 
middle_df = deliveries_df[deliveries_df['phase']=='middle']

middle_df['is_dot_ball'] = middle_df.apply(lambda row: 1 if row['batsman_runs']==0 & row['extra_runs']==0 else 0,axis=1)
batsmen_middle_grouped = middle_df.groupby(['batsman','match_id']).agg({'is_dot_ball':sum,'ball':'count'}).reset_index().rename(columns={'is_dot_ball':'dot_balls','ball':'balls_faced'})

batsmen_middle_grouped['dotball%'] = batsmen_middle_grouped['dot_balls']*100.0/batsmen_middle_grouped['balls_faced']
agg = batsmen_middle_grouped.groupby('batsman').agg({'dotball%':'mean','balls_faced':'sum'}).reset_index().rename(columns={'dotball%':'avg_dot_ball_%','balls_faced':'total_balls_faced'})

agg = agg[agg['total_balls_faced']>=600]

agg = agg.sort_values('avg_dot_ball_%',ascending=True)
g = sns.barplot(x='batsman',y='avg_dot_ball_%',data=agg.iloc[0:10,]);

fig = g.get_figure()

fig.set_size_inches(12,10)

fig.suptitle("Average Dot ball % Top 10 Batsmen in middle overs(played more than 600 balls)")

fig.tight_layout()

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height + 0.3,'{:1.2f}'.format(height),ha="center") 
agg.tail(10)
bowler_middle_grouped = middle_df.groupby(['bowler','match_id']).agg({'is_bowler_wicket':'sum','over':pd.Series.nunique}).reset_index().rename(columns={'is_bowler_wicket':'wickets_taken','over':'overs_bowled'})
agg = bowler_middle_grouped.groupby('bowler').agg({'wickets_taken':'sum','overs_bowled':'sum'}).reset_index().rename(columns={'strikerate':'avg_sr','overs_bowled':'num_overs_bowled'})

agg = agg[agg['num_overs_bowled']>=100]

agg['strikerate'] = agg['num_overs_bowled']*6.0/agg['wickets_taken']

agg = agg.sort_values('strikerate',ascending=True)
agg.tail(10)
g = sns.barplot(x='bowler',y='strikerate',data=agg.iloc[0:10,]);

fig = g.get_figure()

fig.set_size_inches(12,10)

fig.suptitle("Top 10 Bowlers in Middle overs(bowled more than 100 overs)")

fig.tight_layout()

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 
slog_df = deliveries_df[deliveries_df['phase']=='slog/death']
agg = slog_df.groupby(['batsman']).agg({'is_boundary':sum,'ball':'count'}).reset_index().rename(columns={'is_boundary':'num_boundaries','ball':'balls_faced'})

agg = agg[agg['balls_faced']>=200]

agg['big_shot_rate'] = agg['balls_faced']/agg['num_boundaries']

agg = agg.sort_values('big_shot_rate',ascending=True)
agg.tail(10)
g = sns.barplot(x='batsman',y='big_shot_rate',data=agg.iloc[0:10,]);

fig = g.get_figure()

fig.set_size_inches(12,10)

fig.suptitle("Top 10 Batsmen in slog/death overs(180 balls or more)")

fig.tight_layout()

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 
agg = slog_df.groupby(['bowler']).agg({'is_boundary':sum,'ball':'count'}).reset_index().rename(columns={'is_boundary':'num_boundaries','ball':'balls_bowled'})

agg = agg[agg['balls_bowled']>=200]

agg['containing_rate'] = agg['balls_bowled']/agg['num_boundaries']

agg = agg.sort_values('containing_rate',ascending=False)
agg.tail(10)
g = sns.barplot(x='bowler',y='containing_rate',data=agg.iloc[0:10,]);

fig = g.get_figure()

fig.set_size_inches(12,10)

fig.suptitle("Top 10 Bowlers in slog/death overs(40 overs or more)")

fig.tight_layout()

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 