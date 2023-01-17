!pip install -U ppscore
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import ppscore as pps

import warnings

warnings.filterwarnings("ignore")



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"
data = pd.read_csv('../input/ipldata/matches.csv')

top_5_teams = ['Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings', 'Royal Challengers Bangalore', 'Kings XI Punjab']

df = data[['season','winner']].copy().dropna()

df['win_count'] = df.groupby(['season','winner'])['winner'].transform('count')

perc = df.loc[:,["season","winner",'win_count']]

perc = perc.drop_duplicates()

perc = perc.loc[perc['winner'].isin(top_5_teams)]

perc = perc.sort_values("season")

fig=px.bar(perc,x='winner', y="win_count", animation_frame="season", 

           animation_group="winner", color="winner", hover_name="winner")

fig.update_layout(title='Wins by top teams per year', showlegend=False)

fig.show()
data = pd.read_csv('../input/ipldata/deliveries.csv')

match = pd.read_csv('../input/ipldata/matches.csv')
fig = data.nunique().reset_index().plot(kind='bar', x='index', y=0, color=0)

fig.update_layout(title='Unique Value Count Plot', xaxis_title='Variables', yaxis_title='Unique value count')

fig.show()
fig = data.isnull().sum().reset_index().plot(kind='bar', x='index', y=0)

fig.update_layout(title='Missing Value Plot', xaxis_title='Variables', yaxis_title='Missing value count')

fig.show()
df = data['dismissal_kind'].value_counts().reset_index()

df.columns = ['Dismissal Kind', 'Count']

fig = px.pie(df, values='Count', names='Dismissal Kind',title='Most common ways of dismissal in IPL 2019')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
sns.heatmap(data.corr())

plt.title('Correelation in data')

plt.show()
matrix_df = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

plt.figure(figsize=(10,10))

sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

plt.title('PPS Matrix')

plt.show()
df = data.loc[data['is_super_over']==1, ['bowler','is_super_over']].copy()

df['total_balls'] = df.groupby('bowler')['is_super_over'].transform('count')

df = df.drop('is_super_over',axis=1).drop_duplicates().reset_index(drop=True).sort_values('total_balls', ascending=False).head(15)

fig = df.plot(kind='bar', x='bowler', y='total_balls', color='total_balls')

fig.update_layout(title='Bowlers who bowled most balls in super overs')

fig.show()
df = data[['bowler', 'total_runs']].copy()

df['ball_count'] = df.groupby('bowler')['total_runs'].transform('count')

df['total_runs'] = df.groupby('bowler')['total_runs'].transform('sum')

df['runs_per_ball'] = df['total_runs'] / df['ball_count']

df = df.drop_duplicates().reset_index(drop=True).sort_values('runs_per_ball').head(15)

fig = df.plot(kind='bar', y='bowler', x='runs_per_ball', color='runs_per_ball')

fig.update_layout(title='Bowlers with best economy')

fig.show()
df = data[['bowler', 'total_runs']].copy()

df['ball_count'] = df.groupby('bowler')['total_runs'].transform('count')

df['total_runs'] = df.groupby('bowler')['total_runs'].transform('sum')

df['runs_per_ball'] = df['total_runs'] / df['ball_count']

df = df.drop_duplicates().reset_index(drop=True).sort_values('runs_per_ball',ascending=False).head(15)

fig = df.plot(kind='bar', y='bowler', x='runs_per_ball', color='runs_per_ball')

fig.update_layout(title='Bowlers with worst economy')

fig.show()
df = data.loc[data['wide_runs']>0, ['bowler', 'wide_runs']].copy()

df['total_wides'] = df.groupby('bowler')['wide_runs'].transform('count')

df = df.drop('wide_runs', axis=1).drop_duplicates().reset_index(drop=True).sort_values('total_wides',ascending=False).head(15)

fig = df.plot(kind='bar', x='bowler', y='total_wides', color='total_wides')

fig.update_layout(title='Bowlers with most wide balls')

fig.show()
df = data[['batsman', 'total_runs']].copy()

df['total_runs'] = df.groupby('batsman')['total_runs'].transform('sum')

df = df.drop_duplicates().sort_values('total_runs').tail(15).reset_index(drop=True)

fig = df.plot(kind='bar', x='batsman', y='total_runs', color='total_runs')

fig.update_layout(title='Highest runs by batsman')

fig.show()
df = data[['batsman', 'total_runs']].copy()

df['ball_count'] = df.groupby('batsman')['total_runs'].transform('count')

df['total_runs'] = df.groupby('batsman')['total_runs'].transform('sum')

df['Strike Rate'] = (df['total_runs'] / df['ball_count']) * 100

df = df.drop_duplicates().sort_values('Strike Rate').tail(15).reset_index(drop=True)

fig = df.plot(kind='bar', y='batsman', x='Strike Rate', color='Strike Rate')

fig.update_layout(title='Batsmen with best strike rate')

fig.show()
df = data[['batsman', 'total_runs']].copy()

df['ball_count'] = df.groupby('batsman')['total_runs'].transform('count')

df['total_runs'] = df.groupby('batsman')['total_runs'].transform('sum')

df['Strike Rate'] = (df['total_runs'] / df['ball_count']) * 100

df = df[df['Strike Rate']>0].drop_duplicates().sort_values('Strike Rate').head(15).reset_index(drop=True)

fig = df.plot(kind='bar', y='batsman', x='Strike Rate', color='Strike Rate')

fig.update_layout(title='Batsmen with worst strike rate')

fig.show()
df = data.loc[data['is_super_over']==1, ['batsman','total_runs']].copy()

df['total_runs'] = df.groupby('batsman')['total_runs'].transform('sum')

df = df.drop_duplicates().sort_values('total_runs', ascending=False).head(15).reset_index(drop=True)

fig = df.plot(kind='bar', x='batsman', y='total_runs', color='total_runs')

fig.update_layout(title='Batsman who scored most runs in super overs')

fig.show()
df = data[['dismissal_kind','fielder']].copy()

df = df.dropna()

df['dismiss_count'] = df.groupby(['dismissal_kind','fielder'])['fielder'].transform('count')

df = df[df['dismissal_kind'].isin(['caught', 'run out'])]

df = df.drop_duplicates()

df1 = df[df['dismissal_kind']=='caught'].sort_values('dismiss_count').tail(10)

df2 = df[df['fielder'].isin(df1['fielder'].tolist())]

df2 = df2[df2['dismissal_kind']=='run out'].sort_values('dismiss_count').tail(10)

df = pd.concat([df1,df2],axis=0)

df = df.reset_index(drop=True)

fig = df.plot(kind='bar', x='fielder', y='dismiss_count', color='dismissal_kind')

fig.update_layout(title='Best performing fielders')

fig.show()
df = data[['bowling_team','dismissal_kind','fielder']].copy()

df = df.dropna()

df = df[df['dismissal_kind']=='stumped']

df['stump_count'] = df.groupby(['bowling_team','fielder'])['dismissal_kind'].transform('count')

df = df.drop('dismissal_kind', axis=1).drop_duplicates().sort_values('stump_count').tail(20).reset_index(drop=True)

fig = df.plot(kind='bar', x='fielder', y='stump_count', color='bowling_team')

fig.update_layout(title='Best Stumpers and their teams')

fig.show()
data = match.copy()

df = data[['city','winner']].copy().dropna()

df['win_count'] = df.groupby(['city','winner'])['winner'].transform('count')

df = df.drop_duplicates().sort_values('win_count').tail(40)

fig = df.plot(kind='bar', x='winner', y='win_count', color='city')

fig.update_layout(title='Teams with most wins and the cities')

fig.show()
df = data[['toss_winner', 'toss_decision']].copy()

df['count'] = df.groupby(['toss_winner', 'toss_decision'])['toss_decision'].transform('count')

df = df.drop_duplicates().reset_index(drop=True)

fig = df.plot(kind='bar', x='toss_winner', y='count', color='toss_decision', barmode='group')

fig.update_layout(title='Toss wins and decisions')

fig.show()
top_players = data['player_of_match'].value_counts().reset_index()['index'].tolist()[:7]

df = data[['winner','player_of_match']].dropna()

df['POM_count'] = df.groupby(['winner','player_of_match'])['player_of_match'].transform('count')

df = df.drop_duplicates().reset_index(drop=True).sort_values('POM_count')

df = df[df['player_of_match'].isin(top_players)]

fig = df.plot(kind='bar', x='player_of_match', y='POM_count', color='winner')

fig.update_layout(title='Players with most Player of the match titles and their teams')

fig.show()
df = data['result'].value_counts().reset_index()

df.columns = ['result', 'count']

fig = px.pie(df, values='count', names='result',title='Most common results')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()