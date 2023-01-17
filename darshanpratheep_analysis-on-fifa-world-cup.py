import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as tick

import plotly.express as px

import plotly.graph_objects as go

import datetime
wc=pd.read_csv('../input/fifa-world-cup/WorldCupMatches.csv')
wc.head()
wc.tail(10)
wc.isnull().sum()
world_cup=wc.dropna()
world_cup.tail()
df=world_cup[['Year','Stage','Stadium','Home Team Name','Home Team Initials','Home Team Goals','Away Team Goals','Away Team Initials','Away Team Name','Attendance','Half-time Home Goals','Half-time Away Goals']]
column_name={'Home Team Name':'T1_Name','Home Team Initials':'Team_1','Home Team Goals':'T1_Goals_Scored','Away Team Goals':'T2_Goals_Scored',

'Away Team Initials':'Team_2','Away Team Name':'T2_Name','Half-time Home Goals':'T1_Half_Time_Goals','Half-time Away Goals':'T2_Half_Time_Goals'}

df.rename(columns=column_name,inplace=True)
df
no_of_teams=df['T1_Name'].unique()

print(no_of_teams)
len(no_of_teams)
no_of_world_cups_hosted=df['Year'].unique()

len(no_of_world_cups_hosted)
no_of_stadiums=df['Stadium'].unique()
len(no_of_stadiums)
col=df['Attendance']

a=col.max()

a
a=df['T1_Goals_Scored']

b=df['T2_Goals_Scored']

if a.max() > b.max():

    print(a.max())

else:

    print(b.max())
df_att=df.groupby(by='Year')['Attendance'].apply(sum)
fig1=px.line(df_att,title='Analysis on Number of Spectators in World Cup')

fig1.update_layout(xaxis_title='Year',yaxis_title='Number of Spectators(in millions)')

fig1.show()
df['Winner']=0
for i in range (df.index.size): 

    if (df['T1_Goals_Scored'].iloc[i] > df['T2_Goals_Scored'].iloc[i]) :

        df['Winner'].iloc[i]=df['T1_Name'].iloc[i]

    elif (df['T1_Goals_Scored'].iloc[i] < df['T2_Goals_Scored'].iloc[i]):

        df['Winner'].iloc[i]=df['T2_Name'].iloc[i]

    else:

        df['Winner'].iloc[i]='Draw'
df_teams=df.groupby(by='T1_Name')[('T1_Goals_Scored','T2_Goals_Scored')].apply(sum)
df_teams['Total_Goals']=df_teams['T1_Goals_Scored']+df_teams['T2_Goals_Scored']
df_teams.drop(['T1_Goals_Scored','T2_Goals_Scored'],axis=1,inplace=True)
fig2=px.bar(df_teams.sort_values('Total_Goals',ascending=False).head(20))

fig2.update_layout(title='Analysis on Goals Scored by Top 20 Teams',xaxis_title='Teams(Top 20)',yaxis_title='Number of Goals Scored')

fig2.show()
df_matches = df.groupby(by=['Year'])['Stage'].count()
fig3=px.scatter(data_frame=df_matches,y='Stage',trendline='lowess')

fig3.update_layout(title='Number of matches in World Cups',xaxis_title='Year',yaxis_title='Number of Matches')

fig3.show()
df_goals=df.groupby(by='Year')[('T1_Goals_Scored','T2_Goals_Scored')].apply(sum)
df_goals['Total_Goals_Year'] = df_goals['T1_Goals_Scored'] + df_goals['T2_Goals_Scored']
df_goals.drop(['T1_Goals_Scored','T2_Goals_Scored'],axis=1,inplace=True)
fig4=px.bar(data_frame=df_goals)

fig4.update_layout(title='Analysis on Number of Goals Scored in World Cup',xaxis_title='Year',yaxis_title='Number of Goals')

fig4.update_traces(marker=dict(color="Crimson"))

fig4.show()
df_stadium=df.groupby(by='Stadium')['Stadium'].count()
df11=df_stadium.sort_values(ascending=False).head(10)
fig5=px.line(df11)

fig5.update_layout(title='Analysis on Stadium',xaxis_title='Stadium',yaxis_title='Number of Matches Hosted')

fig5.show()
df_88=df.groupby(by=['T1_Name'])[('T1_Name')].count()
df_99=df.groupby(by=['T2_Name'])[('T2_Name')].count()
wc1=df_88 + df_99
wc2 = pd.DataFrame(wc1)

wc2.columns=['Count']
fig6=px.bar(data_frame=wc2.sort_values(('Count'),ascending=False).head(20))

fig6.update_layout(title='Analysis on Number of Goals scored by Top 20 Teams',xaxis_title='Team',yaxis_title='Number of Goals Scored')

fig6.update_traces(marker=dict(color="LightSeaGreen"))

fig6.show()
total_matches=int(wc2.loc['Argentina'].values)
total_matches
win=len(df[df['Winner']=='Argentina'])
win
draw_matches=df[df['Winner']=='Draw']
draw=0

list_team1=list(draw_matches['T1_Name'])

list_team2=list(draw_matches['T2_Name'])

for i in range(draw_matches.index.size):

    if list_team1[i] == 'Argentina':

        draw = draw+1

    elif list_team2[i] == 'Argentina':

        draw = draw+1
draw
loss=total_matches-(win + draw)
loss
ARGENTINA={'WIN':win,'DRAW':draw,'LOSS':loss}

df_argentina=pd.Series(ARGENTINA)
df_argentina
fig7=px.pie(df_argentina,values=df_argentina.values,names=('WIN','DRAW','LOSS'),title='Argentina World Cup Performance',hole=0.3,hover_name=('WIN','DRAW','LOSS'))

fig7.show()