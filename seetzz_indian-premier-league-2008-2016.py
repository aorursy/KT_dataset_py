import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
deliveries = pd.read_csv('../input/deliveries.csv')

deliveries.head(1)
deliveries.groupby(['match_id', 'inning']).agg({'total_runs': 'sum'})[:6]
matches = pd.read_csv('../input/matches.csv')

matches.head(1)
matches.hist(column='season', figsize=(9,6),bins=20)  
matches['city'].value_counts().plot(kind='barh', figsize=(10,8), rot=0)
ipl = matches[['id', 'season']].merge(deliveries, left_on = 'id', right_on ='match_id').drop('match_id', axis = 1)
ipl.head(1)
ipl.shape
ipl.columns
ipl.total_runs.value_counts()
matches.groupby('season').winner.value_counts()
match_winners = matches.winner.value_counts()

fig, ax = plt.subplots(figsize=(8,7))

explode = (0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4)

ax.pie(match_winners, labels = None, autopct='%1.1f%%', startangle=90, shadow = True, explode = explode)

ax.legend(bbox_to_anchor=(1,1), labels=match_winners.index)
ipl.groupby(['inning']).agg({'total_runs': 'sum'})
ipl.groupby(['inning']).agg({'ball': 'count'})
ipl.groupby(['season','inning']).agg({'total_runs': 'sum', 'ball': 'count'})
ipl.groupby(['season','inning']).agg({'total_runs': 'sum'}).plot(kind='barh', figsize=(10,8))

plt.xlabel('Total Runs scored by all teams')

plt.ylabel('Season and Innings')
ipl.groupby(['season']).agg({'total_runs': 'sum'}).plot(kind='bar')
y = ipl.groupby(['season']).agg({'total_runs': 'sum', 'ball': 'count'})

x = matches.groupby(['season']).agg({'id': 'count'})
x2 = x.reset_index()

x2
y2= y.reset_index()

y2
iplz = pd.merge(x2,y2, how='inner', on='season')

ipl2=iplz.set_index('season')

ipl2
ipl2.plot(kind ='barh', colormap = 'plasma')
t = ipl2['total_runs']

b = ipl2['ball']

n = ipl2['id']

tn = (t.T / n.T).T

bn = (b.T / n.T).T

tb = (t.T / b.T).T

z = pd.DataFrame([n, tn,bn,tb])

z.index = ['No.of matches', 'Average Runs per match', 'Average balls bowled per match', 'Average runs per ball']

z.T
z.T.plot(kind='bar', figsize = (12,10), colormap = 'coolwarm')

plt.xlabel('Season')

plt.ylabel('Average')

plt.legend(loc=9,ncol=4)
plt.figure(figsize=(12,8))

t = ipl2['total_runs']

b = ipl2['ball']

n = ipl2['id']

tn = (t.T / n.T).T

bn = (b.T / n.T).T

tb = (t.T / b.T).T

ax = plt.subplot(311) 

plt.plot( tn, 'b.--',)

ax.set_title("Average Runs scored per match")

ax1 = plt.subplot(312)

plt.plot( bn, 'r.--')

ax1.set_title("Average Balls bowled per match")

ax2 = plt.subplot(313)

plt.plot( tb, 'g-.')

ax2.set_title("Average Runs scored per Ball")

plt.subplots_adjust(top=2.0)

plt.tight_layout()



ak = ipl[ipl.batsman.str.lower().str.contains('kohli')].groupby(['season'])['total_runs'].count()

bk = ipl[ipl.batsman.str.lower().str.contains('kohli')].groupby(['season'])['total_runs'].sum()

ck = pd.concat([ak, bk], axis=1)

kohli_strikerate = (bk.T / ak.T*100).T

kohli_strikerate.plot('bar', figsize=(10,8))

plt.xlabel('Season')

plt.ylabel('Virat Kohli Strike Rate (Runs scored per ball)')

plt.title('Virat Kohli in IPL')
deliveries[deliveries["batsman"] == "V Kohli"]["batsman_runs"].value_counts().plot(kind="bar")

plt.title('V Kohli')

plt.ylabel('No of times')

plt.xlabel('Runs')
ipl['batsman'].value_counts()[:10]
# P Kumar's teamwise - runs scored by teams of his bowling

Ash = deliveries[deliveries['bowler'] == 'R Ashwin']  # inning 2

bowler_Ash = Ash.groupby('batting_team')['total_runs'].sum()

bowler_Ash
bowler_Ash.sum()
inning_Ash = Ash.groupby('inning')['total_runs'].sum()

inning_Ash
Run_Ashwin = Ash.groupby('inning')['total_runs'].value_counts()

Run_Ashwin
dismissal_Ash = Ash.groupby('dismissal_kind')['player_dismissed'].count()

dismissal_Ash.plot('bar', rot =50)

plt.ylabel('Number of times')

plt.title('R Ashwin dismissal kind')
all_teams = matches['team1'].unique().tolist() + matches['team2'].unique().tolist()

all_teams = list(set(all_teams))



team_names =[]

played_count = []

won_count = []

for team in all_teams:

    team_names.append(team)

    played = matches[(matches['team1'] == team) | (matches['team2'] == team)].shape[0]

    won = matches[matches['winner'] == team].shape[0]

    

    played_count.append(played)

    won_count.append(won)



data_dict = {

    'team': team_names,

    'played': played_count,

    'won': won_count,   

}



df_played_won = pd.DataFrame(data_dict)

team_won = df_played_won.set_index('team')

team_won.plot(kind='barh', figsize=(10,10))

plt.xlabel('No of times')
season_wins = pd.crosstab(matches.winner, matches.season, margins=True)

season_wins
matches.toss_winner.value_counts()
matches.toss_decision.value_counts()
matches.groupby(['season','toss_decision'])['id'].count().plot('barh', figsize=(10,8))
matches.groupby(['season','result'])['id'].count()
matches.groupby(['toss_winner', 'toss_decision']).id.count()
matches.groupby([ 'toss_decision', 'toss_winner']).id.count()
matches.groupby('winner')['win_by_runs'].agg(np.max).plot(kind = 'barh', figsize=(10,8))
matches.pivot_table( columns='winner', values='win_by_runs', aggfunc='max').T
over = deliveries.groupby('over', as_index=True).agg({"total_runs": "sum"})

over.plot(kind='bar', figsize=(12,8));
over = deliveries.groupby('over', as_index=True).agg({"extra_runs": "sum"})

over.plot(kind='bar', figsize=(12,8));
overm = deliveries.groupby('ball', as_index=True).agg({"total_runs": "sum"})

overm.plot(kind='barh', figsize=(12,8));
ball_df = deliveries['total_runs'].groupby([deliveries['inning'], deliveries['ball']]).mean()

ball_df
tot = ipl.groupby(['batting_team'])['total_runs']

tot.sum().plot('bar')
ipl.groupby(['batting_team','season','inning']).agg({'total_runs': [ 'max', sum],     

                                     'extra_runs': 'sum',

                                                'batsman_runs': 'sum',

                                     'ball': ['count', 'nunique']}).head(9)
ipl.groupby(['batsman']).batsman_runs.sum().sort_values(ascending = False).head(10)
grp = deliveries.groupby('batsman')['batsman_runs'].sum()

grp.reset_index().sort_values(by=['batsman_runs'], ascending = False).head()
grp1 = deliveries.groupby(['bowling_team','bowler'])['player_dismissed'].count()

grp1.reset_index().sort_values(by=['player_dismissed'], ascending = False)[:5]
ipl.groupby(['batsman', 'inning']).batsman_runs.sum().sort_values(ascending = False).head(10)
ipl.player_dismissed.value_counts().head()
ipl.fielder.value_counts().head()
ipl.dismissal_kind.value_counts().head()
deliveries.batsman_runs.value_counts().plot('bar')

plt.xticks(np.arange(0,7, 1), rotation = 40);

plt.yticks(np.arange(0,60000, 4000));
deliveries.groupby(['inning','batting_team']).total_runs.sum()