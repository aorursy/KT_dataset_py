import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.graph_objs import *

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from numpy import median

init_notebook_mode(connected=True)

%matplotlib inline 
ipl_deliveries=pd.read_csv('../input/deliveries.csv')

ipl_matches=pd.read_csv('../input/matches.csv')
print(ipl_deliveries.batting_team.unique())
print("Total number of batsmen are :", len(ipl_deliveries.batsman.unique()))

print("Total number of bowlewr are :", len(ipl_deliveries.bowler.unique()))
print("Number of matches played are :",len(ipl_deliveries.match_id.unique()))
Matches_at_Venu=ipl_matches.groupby('venue').count()[['id']].reset_index()
Matches_at_Venu.head(10)
fig, ax = plt.subplots()

fig.set_size_inches(11, 8)

sns.pointplot(y='venue',x='id',data=Matches_at_Venu,figsize=(20,20),linestyles="--",markers='o',color='r')

ax.set_xlabel("number of matches played")

ax.set_ylabel("Stadium")
ipm1=ipl_matches.groupby(by='winner').sum()[['win_by_runs','win_by_wickets']]
ipm1.plot(kind='bar',figsize=(12,8),grid=True,title='Team Score in all seasons')
# number of players dissmissed by a bowler

ipd1=ipl_deliveries.groupby(by='bowler').count()[['player_dismissed']]

#total number of runs given away by the bowler

ipd2=ipl_deliveries.groupby(by='bowler').sum()[['total_runs']]

#number of matches played by each player

ipd3=pd.DataFrame(ipl_deliveries.groupby(by='bowler')[['match_id']].nunique())

bowler=pd.concat([ipd1,ipd2,ipd3],axis=1)
bowler=bowler.sort_values('match_id',ascending=False).reset_index()
bowler.head()
from mpl_toolkits.mplot3d import Axes3D
Z=(bowler.total_runs/(bowler.match_id*6))
#fig=plt.figure(figsize=(10,12))

#ax=fig.add_subplot(111,projection='3d',facecolor='c')

#ax.plot(xs=bowler.player_dismissed,ys=bowler.total_runs)

#ax.plot_wireframe(X=bowler.player_dismissed,Y=bowler.total_runs,Z=(bowler.total_runs/(bowler.match_id*6)),rcount=1000)

#ax.set_xlabel('Players_dismissed in all IPL seasons')

#ax.set_ylabel('Runs given away by bowlers')

#ax.set_zlabel('Bowling average throughout the seasons')
fig=plt.figure(figsize=(10,12))

plt.subplot(polar=True)

sns.countplot(x='city',data=ipl_matches,palette='gist_earth')
fig=plt.figure(figsize=(10,12))

plt.subplot(polar=False)

sns.countplot(y='venue',hue='city',data=ipl_matches,palette='seismic',linewidth=5,edgecolor=sns.color_palette("summer",25))

total_win=ipl_matches.groupby(['season','winner']).count()[['id']].reset_index()
(sns.jointplot(x='season',y='id',data=total_win,size=10,ratio=5,color='m').plot_joint(sns.kdeplot,zborder=0,n_level=6)).set_axis_labels("Season", "Matches won")
sns.factorplot(data=total_win,x='season',y='id',col='winner',col_wrap=3,size=2,kind='bar',aspect=2,saturation=2,

margin_titles=True)
team_stats=pd.DataFrame({'TotalMatches': ipl_matches.team1.value_counts()+ipl_matches.team2.value_counts()

                         ,'TotalWin':ipl_matches.winner.value_counts()})



team_stats=team_stats.reset_index()



team_stats.rename(columns={'index':'Teams'},inplace=True)
team_stats.head(10)
trace_TMatch = Bar(x=team_stats.Teams,

                  y=team_stats.TotalMatches,

                  name='Total Matches Played',

                  marker=dict(color='#ffcdd2'))



trace_WMatch = Bar(x=team_stats.Teams,

                y=team_stats.TotalWin,

                name='Matches Won',

                marker=dict(color='#A2D5F2'))



data = [trace_TMatch, trace_WMatch]

layout = Layout(title="Win vs Los comparison for each team",

                xaxis=dict(title='Teams'),

                yaxis=dict(title='Number of Matches '))

fig = Figure(data=data, layout=layout)



iplot(fig,filename='C:/Lectures/DataVisualization/ipl/stackbar')
# The final match of the year is the one we need so we remove all the duplicates from the data 

# and only keep the last row of the subset

season_winner=ipl_matches.drop_duplicates(subset=['season'], keep='last')[['season','winner']].reset_index(drop=True)

season_winner
fig=plt.figure(figsize=(8,5))

plt.subplot()

sns.countplot(y='winner',data=season_winner,palette='coolwarm')