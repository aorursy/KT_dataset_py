from IPython.display import Image

import os

!ls ../input/

Image("../input/image-ipl/ipl.jpg")


import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import seaborn as sns

%matplotlib inline
d_matches=pd.read_csv('../input/ipl20082019/matches.csv')

d_deliveries=pd.read_csv('../input/ipl20082019/deliveries.csv')

d_teams=pd.read_csv('../input/ipl20082019/teams.csv')

d_team_hna=pd.read_csv('../input/ipl20082019/teamwise_home_and_away.csv')


d_matches.head()
d_deliveries.head()
d_teams
d_team_hna
(d_matches.isna().sum()/len(d_matches))*100
d_matches.drop(labels='umpire3',axis=1,inplace=True)
d_matches[d_matches.city.isnull()]
d_matches.loc[d_matches.venue=='Dubai International Cricket Stadium']

d_matches.city.fillna('Dubai',inplace=True) 
d_matches.dropna(axis=0,subset=['winner','player_of_match'],inplace=True)
d_matches.dropna(axis=0,subset=['umpire1','umpire2'],inplace=True)
(d_deliveries.isna().sum()/len(d_deliveries))*100
d_deliveries.drop(axis=1,columns=['player_dismissed','dismissal_kind','fielder'])
(d_teams.isnull().sum()/len(d_teams))*100
(d_team_hna.isnull().sum()/len(d_team_hna))*100
px.bar(d_matches.groupby(by='city')[['date']].count(),text='value',color_discrete_sequence= ['seagreen'],labels={'value':'No.of days'},title='No.of matches played in each city')
d_matches.toss_winner==d_matches.winner

df=d_matches[(d_matches.toss_winner==d_matches.winner)].groupby('winner')[['toss_winner']].count().sort_values('toss_winner')

px.line(df,x=df.index,y=df.toss_winner.values,color_discrete_sequence=['red'],labels={'winner':'teams','y':'Matches won'},title='Matches won by teams after winning the toss')
px.pie(d_team_hna,names='team',values='home_wins',color_discrete_sequence=px.colors.sequential.Plasma_r,title='Percentage of home wins')
px.line(d_team_hna.groupby('team')[['home_wins','away_wins']].sum(),color_discrete_sequence=['darkorchid','magenta'],labels={'value':'No. of wins'},title='Comparison of wins')
d=d_matches.loc[d_matches.Season=='IPL-2019'].groupby('winner')[['winner']].count()

x=d.index

y=d.winner.values

f1=px.bar(data_frame=d,x=d.index,y=d.winner.values,color_discrete_sequence=['goldenrod'],labels={'index':'teams','y':'No.of wins'},title='Wins by each team in 2019')

f1.show()
d=d_matches.pivot_table(index=['winner'],values=['win_by_runs','win_by_wickets'],aggfunc=sum)

d.plot(kind='box',figsize=(30,10))

plt.title('Comaprison of wins by runs and wins by wickets')

plt.show()

df=pd.pivot_table(data=d_team_hna,values=['home_matches','away_matches'],index='team',aggfunc='sum').sort_values(by=['home_matches','away_matches'])

fig = make_subplots(rows=1, cols=2)



fig.add_trace(go.Scatter(x=df.index, y=df.home_matches.values,name='Home Matches'),

    row=1, col=1)



fig.add_trace(

    go.Scatter(x=df.index, y=df.away_matches.values,name='Away Matches'),

    row=1, col=2)



fig.update_layout(height=600, width=800, title_text="Comparison of matches played in home city and away")

fig.show()