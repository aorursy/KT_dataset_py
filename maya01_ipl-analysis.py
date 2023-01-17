# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/ipldata/'
deliveries = pd.read_csv(path+'deliveries.csv')
matches = pd.read_csv(path+'matches.csv')
#Replacing the Team Names with their abbreviations

matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab','Delhi Capitals',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DCS','CSK','RR','DD','GL','KXIP','DC','SRH','RPS','KTK','PW','RPS'],inplace=True)

deliveries.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab','Delhi Capitals',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DCS','CSK','RR','DD','GL','KXIP','DC','SRH','RPS','KTK','PW','RPS'],inplace=True)
print('Shape Deliveries: ',deliveries.shape)
print('Shape Matches: ',matches.shape)
matches.head(2)
matches.drop(['umpire3'],axis=1,inplace=True)  #since all the values are NaN
deliveries.fillna(0,inplace=True)
print('Total number of IPL-Seasons played till now: ',matches.season.nunique())
print('List of Teams played in IPL till now:\n ',np.unique(matches[['team1','team2']].values))
print('First IPL match :\n')
x= matches[matches.season==matches.season.min()]
one_ipl = x[x.date==x.date.min()]
one_ipl[['season','city','date','team1','team2','winner','win_by_runs','player_of_match']].style.background_gradient(cmap='viridis')
print('Last IPL match :\n')
x= matches[matches.season==matches.season.max()]
#last_ipl = x[x.date==x.date.max()]
x[['season','city','date','team1','team2','winner','win_by_runs','player_of_match']].tail(1).style.background_gradient(cmap='viridis')
print('List of Cities where IPL happened :',list(matches.city.unique()))
plt.figure(figsize=(12,8))
plt.title('Matches played at different cities')
sns.countplot(y=matches.city,orient='h',order = matches.city.value_counts().index)
plt.show()
plt.figure(figsize=(12,6))
plt.title('Top 10 MAN OF THE MATCH getter')
ax=sns.countplot(x=matches.player_of_match,order = matches.player_of_match.value_counts()[:10].index,palette="Set3")
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()
plt.figure(figsize=(12,6))
plt.title('Matches Won by the Teams')
ax=sns.countplot(x=matches['winner'],order = matches['winner'].value_counts().index,palette=sns.dark_palette((260, 75, 60), input="husl"))

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()
matches_played_by_teams = pd.concat([matches.team1,matches.team2])
matches_played_by_teams = matches_played_by_teams.value_counts().reset_index()
matches_played_by_teams.columns = ['Team','Total Matches']
wins_by_teams = matches.winner.value_counts().reset_index()
wins_by_teams.columns = ['Team','wins']
merged_team= wins_by_teams[['Team','wins']].merge(matches_played_by_teams, left_on = 'Team', right_on = 'Team', how = 'right')
merged_team['win %'] =  round((merged_team.wins/merged_team['Total Matches'])*100,2)
merged_team.style.background_gradient(cmap='Greens')
trace1 = go.Bar(
    x=merged_team.Team,
    y=merged_team['Total Matches'],
    name='Total Matches'
)
trace2 = go.Bar(
    x=merged_team.Team,
    y=merged_team['wins'],
    name='Matches Won'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
b = merged_team[['Team','win %']]
b=b.sort_values(by='win %',ascending=False)

plt.subplots(figsize=(12,6))
ax = sns.barplot(y='win %',x='Team',data=b,palette=sns.color_palette('RdYlGn',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()
max_win_by_wickets = matches[matches['win_by_wickets']==matches.win_by_wickets.max()]
max_win_by_wickets[['season','team1','team2','winner','win_by_wickets','player_of_match']].style.background_gradient(cmap='Blues')
max_win_by_runs = matches[matches['win_by_runs']==matches.win_by_runs.max()]
max_win_by_runs[['season','team1','team2','winner','win_by_runs','player_of_match']].style.background_gradient(cmap='Set3')
plt.subplots(figsize=(10,6))
sns.countplot(x='season',hue='toss_decision',data=matches)
plt.show()
toss_decision = matches.toss_decision.value_counts(normalize=True)
plt.pie(toss_decision,labels=toss_decision.index,autopct='%1.1f%%',)
plt.axis('equal')
plt.tight_layout()
plt.show()
match_toss = pd.concat([matches.team1,matches.team2])
match_toss = match_toss.value_counts().reset_index()
match_toss.columns = ['Team','Total Matches']
toss_wins_by_teams = matches.toss_winner.value_counts().reset_index()
toss_wins_by_teams.columns = ['Team','toss_wins']
merged_team_toss= toss_wins_by_teams[['Team','toss_wins']].merge(match_toss, left_on = 'Team', right_on = 'Team', how = 'right')
merged_team_toss['toss_win_%'] =  round((merged_team_toss.toss_wins/merged_team_toss['Total Matches'])*100,2)
temp = matches[matches.toss_winner== matches.winner]
temp = temp.winner.value_counts().reset_index()
temp.columns=['Team','match_wins']
merged_team_toss_wins= temp[['Team','match_wins']].merge(merged_team_toss, left_on = 'Team', right_on = 'Team', how = 'right')
merged_team_toss_wins['match_win_%'] = round((merged_team_toss_wins.match_wins/merged_team_toss_wins['toss_wins'])*100,2)
merged_team_toss_wins[['Team','Total Matches','toss_wins','match_wins','toss_win_%','match_win_%']].style.background_gradient(cmap='Greens')
plt.subplots(figsize=(12,6))
ax = sns.countplot(x=matches.toss_winner,order = matches['toss_winner'].value_counts().index,palette=sns.color_palette('RdYlGn',50))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()
plt.subplots(figsize=(12,6))
ax = sns.barplot(y='toss_win_%',x='Team',data=merged_team_toss,palette=sns.color_palette('GnBu',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

plt.subplots(figsize=(12,6))
plt.title('Winning % of Teams if they Wins the Toss ')
ax = sns.barplot(y='match_win_%',x='Team',data=merged_team_toss_wins,palette=sns.color_palette('colorblind',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()
print('Total number of MATCHES PLAYED so far :',len(deliveries['match_id'].unique()))
plt.subplots(figsize=(10,6))
plt.title('Matches played across each seasons')
ax = sns.countplot(x='season',data=matches,palette=sns.color_palette('Paired',80))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()
matches_not_normal = matches[matches.result !='normal']
matches_not_normal.result.value_counts()
matches_nr = matches_not_normal[matches_not_normal['result']=='no result']
matches_nr[['season','team1','team2','result']].style.background_gradient(cmap='pink')
matches_tie = matches_not_normal[matches_not_normal['result']=='tie']
matches_tie[['season','team1','team2','result']].style.background_gradient(cmap="YlGn")
matches_dl = matches[matches.dl_applied !=0]
print('Total Number of DL affected Matches :',matches_dl.shape[0],'\n list :\n')
matches_dl[['season','team1','team2','winner','player_of_match']].style.background_gradient(cmap="PRGn")
def season_by_player(season):
  season_player = matches[matches.season==season]
  season_player = Counter(season_player.player_of_match)
  season_player = pd.DataFrame.from_dict(season_player,orient='index').reset_index()
  season_player.columns = ['player','counts']
  season_player = season_player.sort_values(by='counts',ascending=False)
  plt.subplots(figsize=(16,6))
  plt.title('Most Player-of-Match in '+str(season))
  ax = sns.barplot(y='counts',x='player',data=season_player[:10],palette=sns.color_palette('colorblind',20))
  for p in ax.patches:
      ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
  plt.show()
season_by_player(2019)
print('Total number of DELIVERIES BOWLED so far :',deliveries.shape[0])
print('Total RUNS SCORED by the batsman so far :',deliveries.total_runs.sum())
print('Total number of WIDE RUNS so far :',deliveries.wide_runs.sum())
print('Total number of NO-BALL RUNS so far :',deliveries.noball_runs.sum())
print('Total number of BYE RUNS so far :',deliveries.bye_runs.sum())
print('Total number of LEG-BYE RUNS so far :',deliveries.legbye_runs.sum())
print('Total number of PENALTY RUNS so far :',deliveries.penalty_runs.sum())
print('Total number of BATSMAN RUNS so far :',deliveries.batsman_runs.sum())
total_extra_runs = deliveries.wide_runs.sum()+deliveries.noball_runs.sum()+deliveries.bye_runs.sum()+deliveries.legbye_runs.sum()+deliveries.penalty_runs.sum()
print('Total EXTRA RUNS scored by the batsman so far :',total_extra_runs)
print('Total number of WICKET FALLED so far :',deliveries[deliveries['player_dismissed']!=0].shape[0])
print('Total SUPER OVERS so far :',deliveries[deliveries.is_super_over==1].match_id.nunique())
runs_by_team = deliveries.groupby(by='batting_team')['batting_team','total_runs'].sum()
runs_by_team = runs_by_team.sort_values('total_runs',ascending=False).reset_index()
runs_by_team.style.background_gradient(cmap='Purples')
runs_by_team = deliveries.groupby(by='batting_team')['batting_team','total_runs'].sum().sort_values('total_runs',ascending=True)
ax = runs_by_team.plot.barh(width=.9,color=sns.color_palette('inferno'))
ax.set_xlabel('Total Runs made by Batting Team till now')
ax.set_ylabel('Teams')
plt.show()
runs_by_team_bowl = deliveries.groupby(by='bowling_team')['bowling_team','total_runs'].sum()
runs_by_team_bowl = runs_by_team_bowl.sort_values('total_runs',ascending=False).reset_index()
runs_by_team_bowl.style.background_gradient(cmap="Greens")
runs_by_team_bowl = deliveries.groupby(by='bowling_team')['bowling_team','total_runs'].sum().sort_values('total_runs',ascending=True)
ax = runs_by_team_bowl.plot.barh(width=.9,color=sns.color_palette("ch:2.5,-.2,dark=.3"))
ax.set_xlabel('Total Runs Given by Bowling Teams till now')
ax.set_ylabel('Teams')
plt.show()
extra_by_team = deliveries.groupby('bowling_team')['bowling_team','extra_runs'].sum().sort_values('extra_runs',ascending=False).reset_index()
extra_by_team.style.background_gradient(cmap="OrRd")
extra_by_team = deliveries.groupby('bowling_team')['bowling_team','extra_runs'].sum().sort_values('extra_runs',ascending=True)
ax = extra_by_team.plot.barh(width=.9,color=sns.color_palette('OrRd'))
ax.set_xlabel('Total Extra Runs given by the Teams till now')
ax.set_ylabel('Teams')
plt.show()
wide_by_team = deliveries.groupby('bowling_team')['bowling_team','wide_runs'].sum().sort_values('wide_runs',ascending=False).reset_index()
wide_by_team.style.background_gradient(cmap="YlGn")
wide_by_team = deliveries.groupby('bowling_team')['bowling_team','wide_runs'].sum().sort_values('wide_runs',ascending=True)
ax = wide_by_team.plot.barh(width=.9,color=sns.color_palette('YlGn'))
ax.set_xlabel('Total Wides given by the Teams till now')
ax.set_ylabel('Teams')
plt.show()
noball_by_team = deliveries.groupby('bowling_team')['bowling_team','noball_runs'].sum().sort_values('noball_runs',ascending=False).reset_index()
noball_by_team.style.background_gradient(cmap="PRGn")
noball_by_team = deliveries.groupby('bowling_team')['bowling_team','noball_runs'].sum().sort_values('noball_runs',ascending=True)
ax = noball_by_team.plot.barh(width=.9,color=sns.color_palette('PRGn'))
ax.set_xlabel('Total No-ball given by the Teams till now')
ax.set_ylabel('Teams')
plt.show()
legbye_by_team = deliveries.groupby('bowling_team')['bowling_team','legbye_runs'].sum().sort_values('legbye_runs',ascending=False).reset_index()
legbye_by_team.style.background_gradient(cmap="PuOr")
legbye_by_team = deliveries.groupby('bowling_team')['bowling_team','legbye_runs'].sum().sort_values('legbye_runs',ascending=True)
ax = legbye_by_team.plot.barh(width=.9,color=sns.color_palette('PuOr'))
ax.set_xlabel('Total Leg-Byes given by the Teams till now')
ax.set_ylabel('Teams')
plt.show()
bye_by_team = deliveries.groupby('bowling_team')['bowling_team','bye_runs'].sum().sort_values('bye_runs',ascending=False).reset_index()
bye_by_team.style.background_gradient(cmap="RdBu")
bye_by_team = deliveries.groupby('bowling_team')['bowling_team','bye_runs'].sum().sort_values('bye_runs',ascending=True)
ax = bye_by_team.plot.barh(width=.9,color=sns.color_palette('RdBu'))
ax.set_xlabel('Total Bye Runs given by the Teams till now')
ax.set_ylabel('Teams')
plt.show()
penalty_by_team = deliveries.groupby('bowling_team')['bowling_team','penalty_runs'].sum().sort_values('penalty_runs',ascending=False).reset_index()
penalty_by_team.style.background_gradient(cmap="RdGy")
penalty = deliveries[deliveries.penalty_runs !=0]
penalty

wickets_by_team=deliveries[deliveries['player_dismissed'] !=0].groupby('bowling_team')['player_dismissed'].count()
wickets_by_team = wickets_by_team.sort_values(ascending=False).reset_index()
wickets_by_team.style.background_gradient(cmap='Reds')
wickets_by_team=deliveries[deliveries['player_dismissed'] !=0].groupby('bowling_team')['player_dismissed'].count().sort_values(ascending=True)
ax = wickets_by_team.plot.barh(width=.9,color=sns.color_palette('Reds'))
ax.set_xlabel('Total Wickets by Teams till now')
ax.set_ylabel('Teams')
plt.show()
ipl_dismissal_kind = deliveries[deliveries.dismissal_kind !=0]
ipl_dismissal_kind = ipl_dismissal_kind['dismissal_kind'].value_counts().plot(kind='barh')
merged_deliveries= matches[['id','season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
merged_deliveries.head(3)
x = merged_deliveries.groupby('batting_team')['wide_runs','bye_runs','legbye_runs','noball_runs','penalty_runs','batsman_runs','extra_runs','total_runs'].sum()
x.style.background_gradient(cmap='Set1')
trace1 = go.Bar(
    x=x.index,
    y=x['wide_runs'],
    name='wide_runs'
)
trace4 = go.Bar(
    x=x.index,
    y=x['bye_runs'],
    name='bye_runs'
)
trace5 = go.Bar(
    x=x.index,
    y=x['legbye_runs'],
    name='legbye_runs'
)
trace6 = go.Bar(
    x=x.index,
    y=x['noball_runs'],
    name='noball_runs'
)
trace7 = go.Bar(
    x=x.index,
    y=x['penalty_runs'],
    name='penalty_runs'
)
trace2 = go.Bar(
    x=x.index,
    y=x['batsman_runs'],
    name='batsman_runs'
)

data = [trace1, trace2,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
def run_distribution_by_team(team):
  z = merged_deliveries[merged_deliveries.batsman_runs !=0]
  zz = z[z['batting_team']==team]
  zzz = zz.groupby('batsman_runs')['batsman_runs'].count()
  zzz = zzz.to_frame()
  zzz.columns = ['r_count']
  fig = px.pie(values=zzz['r_count'],names=list(zzz.index),title='Distribution of Runs for '+team+' across IPL')
  fig.show()
run_distribution_by_team('CSK')
run_distribution_by_team('RCB')
season_runs=merged_deliveries.groupby(['season'])['total_runs'].sum().reset_index()
season_runs.set_index('season').plot(marker='o')
plt.gcf().set_size_inches(10,6)
plt.title('Total Runs Across the Seasons')
plt.show()
temp = merged_deliveries.groupby(['season','match_id'])['total_runs'].sum().reset_index()#.drop('match_id')
temp.columns = ['season','match_id','match_avg_runs']
temp = temp.groupby('season')['match_avg_runs'].mean().reset_index()
cm = plt.cm.get_cmap('hot')
temp.set_index('season').plot(marker='o',colormap='Blues_r')
plt.gcf().set_size_inches(10,6)
plt.title('Average Runs per match Across the Seasons')
plt.show()
matches_team = merged_deliveries.groupby('batting_team')['match_id'].nunique().reset_index()
runperover = merged_deliveries.groupby(['batting_team','over'])['total_runs'].sum().reset_index()
runperover=runperover.merge(matches_team,left_on='batting_team',right_on='batting_team',how='outer')
runperover['run_rate'] = runperover.total_runs/runperover.match_id
run_per_over = runperover[['batting_team','over','run_rate']].set_index(['batting_team','over'])
r = run_per_over.unstack(level=0)
team = matches_team.batting_team
color = ['#ffff14','#3581e6','#664bab','#c93c34','#ff4d2e','#4d115c','#25032e','#ff0d1d','#0f23ba','#44c977','#eb4034','#4498c9','#c9448d','#ff8400']
r.run_rate[team].plot(color=color) #plotting graphs for teams that have played more than 100 matches
x=r.index
plt.xticks(x)
plt.ylabel('Run Rates')
fig=plt.gcf()
fig.set_size_inches(16,10)
plt.show()
high_wickettaker_all = merged_deliveries[merged_deliveries.player_dismissed != 0]

high_wickettaker_all = high_wickettaker_all[(high_wickettaker_all.dismissal_kind !='retired hurt') & (high_wickettaker_all.dismissal_kind !='obstructing the field'
) & (high_wickettaker_all.dismissal_kind !='run out') ]#,'obstructing the field','run out']]
high_wickettaker = high_wickettaker_all.bowler.value_counts().reset_index()
high_wickettaker = pd.DataFrame(high_wickettaker)
high_wickettaker.columns = ['bowler','counts']
high_wickettaker = high_wickettaker.sort_values(by='counts',ascending=False)
plt.subplots(figsize=(16,6))
plt.title('Top 10 highest wicket-taker across IPL')
ax = sns.barplot(y='counts',x='bowler',data=high_wickettaker[:10],palette=sns.color_palette('colorblind',20))
for p in ax.patches:
     ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
     plt.show()
def purple_cap(season):
  purple_cap = high_wickettaker_all[high_wickettaker_all.season==season]
  purple_cap = purple_cap.bowler.value_counts().reset_index()
  purple_cap.columns = ['bowler','wicket']
  #purple_cap.head(5)
  plt.subplots(figsize=(16,4))
  plt.title('Top 5 PURPLE CAP contendors in IPL-'+str(season))
  values = purple_cap.wicket[:5]
  print(str(list(purple_cap.bowler[:1]))+' won the PURPLE CAP in IPL- ',str(season))
  clrs = ['grey' if (x < max(values)) else 'purple' for x in values ]
  ax=sns.barplot(x='bowler', y='wicket',data=purple_cap[:5], palette=clrs) # color=clrs)
  for p in ax.patches:
      ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
      plt.show()
purple_cap(2019)
highest_run_scorer = merged_deliveries.groupby('batsman')['batsman_runs'].sum().reset_index()
highest_run_scorer = highest_run_scorer.sort_values('batsman_runs',ascending=False)

plt.subplots(figsize=(16,8))
plt.title('Top 10 RUN-MACHINES across IPL')
ax = sns.barplot(y='batsman_runs',x='batsman',data=highest_run_scorer[:10],palette=sns.color_palette('bright',20))
for p in ax.patches:
     ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
     plt.show()
def orange_cap(season):
  orange_cap = merged_deliveries[merged_deliveries.season==season]
  orange_cap = orange_cap.batsman.value_counts().reset_index()
  orange_cap.columns = ['batsman','runs']
  #print(orange_cap.head(5))
  plt.subplots(figsize=(16,4))
  plt.title('Top 5 ORANGE CAP contendors in IPL-'+str(season))
  values = orange_cap.runs[:5]
  print(str(str(orange_cap.batsman[:1].values))+' won the ORANGE CAP in IPL- ',season)
  clrs = ['grey' if (x < max(values)) else 'Orange' for x in values ]
  ax=sns.barplot(x='batsman', y='runs',data=orange_cap[:5], palette=clrs) # color=clrs)
  for p in ax.patches:
      ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
      plt.show()
orange_cap(2019)
innings = deliveries.groupby('batsman')['match_id'].nunique().reset_index()
bat=deliveries.groupby(['batsman'])['ball'].count().reset_index()
runs=deliveries.groupby(['batsman'])['batsman_runs'].sum().reset_index()
bat = innings.merge(bat,left_on='batsman',right_on='batsman',how='outer')
bat=bat.merge(runs,left_on='batsman',right_on='batsman',how='outer')
bat.rename({'match_id':'Innings','ball':'ball_x','batsman_runs':'ball_y'},axis=1,inplace=True)
sixes=deliveries.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index()
fours=deliveries.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index()
bat['strike_rate']=np.round(bat['ball_y']/bat['ball_x']*100,2)
bat['bat_average'] = np.round((bat.ball_y/bat.Innings),2)
bat=bat.merge(sixes,left_on='batsman',right_on='batsman',how='outer')
bat=bat.merge(fours,left_on='batsman',right_on='batsman',how='outer')
compare=deliveries.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()
compare=compare.groupby(['batsman'])['batsman_runs'].max().reset_index()
bat=bat.merge(compare,left_on='batsman',right_on='batsman',how='outer')
bat.rename({'ball_x':'balls','ball_y':'runs','batsman_runs_x':"6's",'batsman_runs_y':"4's",'batsman_runs':'Highest_score'},axis=1,inplace=True)
bat[1:].sort_values('runs',ascending=False).head(10)
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
ct=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
#ct.replace([NaN,nan],[0,0],inplace=True)
bbm = ct.groupby(['match_id','bowler'])['player_dismissed'].count().reset_index()
bbm = bbm.sort_values('player_dismissed',ascending=True)
#bbm
bowl_over = deliveries.groupby(['match_id','bowler'])['total_runs'].sum().reset_index()
bowl_over = bowl_over.sort_values('total_runs',ascending=False)
bowl_wicket_over = bbm.merge(bowl_over,left_on=['match_id','bowler'],right_on=['match_id','bowler'],how='outer')
bf = bowl_wicket_over.groupby(['bowler']).max().reset_index()
bf = bf[['bowler','player_dismissed','total_runs']]

bf['player_dismissed']=bf['player_dismissed'].fillna(0)
bf['player_dismissed']=bf['player_dismissed'].astype(int)
bf['bbm'] = bf[['player_dismissed', 'total_runs']].astype(str).apply(lambda x: ' / '.join(x), axis=1)
bf.replace('NaN', np.NaN)
bfs = bf[['bowler','bbm']]
four_wicket=bowl_wicket_over.groupby('bowler')['player_dismissed'].agg(lambda x: (x==4).sum()).reset_index()
four_wicket.columns = ['bowler','4W']
fifer = bowl_wicket_over.groupby('bowler')['player_dismissed'].agg(lambda x: (x>4).sum()).reset_index()
fifer.columns = ['bowler','5W']
innings = deliveries.groupby('bowler')['match_id'].nunique().reset_index()
balls=deliveries.groupby(['bowler'])['ball'].count().reset_index()
runs=deliveries.groupby(['bowler'])['total_runs'].sum().reset_index()
balls = innings.merge(balls,left_on='bowler',right_on='bowler',how='outer')
balls=balls.merge(runs,left_on='bowler',right_on='bowler',how='outer')
balls.rename({'match_id':'Innings','ball':'ball_x','total_runs':'runs_given'},axis=1,inplace=True)
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
ct=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
wickets = ct.groupby('bowler')['player_dismissed'].count().reset_index()
balls=balls.merge(wickets,left_on='bowler',right_on='bowler',how='outer')
balls['strike_rate']=np.round((balls.ball_x/balls.player_dismissed),2)
balls['average'] = np.round((balls.runs_given/balls.player_dismissed),2)
balls['economy']=np.round(balls['runs_given']/(balls['ball_x']/6),2)
balls=balls.merge(bfs,left_on='bowler',right_on='bowler',how='outer')
balls=balls.merge(four_wicket,left_on='bowler',right_on='bowler',how='outer')
balls=balls.merge(fifer,left_on='bowler',right_on='bowler',how='outer')
balls[['player_dismissed','strike_rate','average','4W','5W']]=balls[['player_dismissed','strike_rate','average','4W','5W']].fillna(0)
balls[['player_dismissed','4W','5W']]=balls[['player_dismissed','4W','5W']].astype(int)
balls.rename({'ball_x':'balls','player_dismissed':'wickets'},axis=1,inplace=True)
balls = balls.sort_values('wickets',ascending=False)
balls.head(10)
ipl_best_bowl = bf[bf.player_dismissed==bf.player_dismissed.max()] 
ipl_best_bowl = ipl_best_bowl[ipl_best_bowl.total_runs ==ipl_best_bowl.total_runs.min()]
ipl_best_bowl [['bowler','bbm']]
most_fifer = balls.sort_values(by='5W',ascending=False)
most_fifer.head()
most_4W = balls.sort_values(by='4W',ascending=False)
most_4W.head()
balls.loc[balls['bowler']=='B Kumar']
best_bowl_avg = balls.loc[balls['wickets']>50].sort_values('average',ascending=True).head(5)
best_bowl_avg
caught = deliveries[deliveries.dismissal_kind=='caught']
caught = caught.groupby('fielder')['bowler'].count().reset_index()
caught.columns = ['feilder','catches']
caught = caught.sort_values('catches',ascending=False)
caught.head(5)
plt.subplots(figsize=(16,4))
plt.title('Most Catches across IPL')
ax = sns.barplot(y='catches',x='feilder',data=caught[:5],palette=sns.color_palette('dark',20))
for p in ax.patches:
     ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
     plt.show()
runouts = deliveries[deliveries.dismissal_kind=='run out']
runouts = runouts.groupby('fielder')['dismissal_kind'].count().reset_index()
runouts.columns = ['fielder','runout']
runouts = runouts.sort_values('runout',ascending=False)
runouts.head(5)
plt.subplots(figsize=(16,4))
plt.title('Most RunOuts by feilder across IPL')
ax = sns.barplot(y='runout',x='fielder',data=runouts[:5],palette=sns.color_palette('dark',20))
for p in ax.patches:
     ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
     plt.show()
bat_runouts = deliveries[deliveries.dismissal_kind=='run out']
bat_runouts = bat_runouts.groupby('player_dismissed')['dismissal_kind'].count().reset_index()
bat_runouts.columns = ['batsman','runout']
bat_runouts = bat_runouts.sort_values('runout',ascending=False)
bat_runouts.head(5)
plt.subplots(figsize=(16,4))
plt.title('Most Run-out faced by any batsman in IPL')
ax = sns.barplot(y='runout',x='batsman',data=bat_runouts[:5],palette=sns.color_palette('dark',20))
for p in ax.patches:
     ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
     plt.show()
