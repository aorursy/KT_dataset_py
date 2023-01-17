# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install mplcyberpunk
import pandas as pd
import numpy as np
import seaborn as sn
import mplcyberpunk
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots
import warnings
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns',None)
df_deliveries = pd.read_csv('/kaggle/input/ipl-data-set/deliveries.csv')
df_players = pd.read_excel('/kaggle/input/ipl-data-set/Players.xlsx')
df_matches = pd.read_csv('/kaggle/input/ipl-data-set/matches.csv')
df_teams = pd.read_csv('/kaggle/input/ipl-data-set/teams.csv')
df_mostruns = pd.read_csv('/kaggle/input/ipl-data-set/most_runs_average_strikerate.csv')
df_homeandaway = pd.read_csv('/kaggle/input/ipl-data-set/teamwise_home_and_away.csv')

df_list = [df_deliveries,df_players,df_matches,df_teams,df_mostruns,df_homeandaway]
name_list = ['df_deliveries','df_players','df_matches','df_teams','df_mostruns','df_homeandaway']
for i,j in zip(df_list,name_list):
    print(f' shape of the dataset <{j}> is <{i.shape}>')
    print('='*100)
plt.figure(figsize = (25,12.5))
plt.style.use('cyberpunk')
sn.relplot(x = 'batsman',y ='total_runs',data=df_mostruns.head(5),kind='line',color = 'yellow')
mplcyberpunk.add_glow_effects()
mplcyberpunk.make_lines_glow()
px.bar(data_frame=df_mostruns.head(10),x = 'batsman',y = 'total_runs',color='batsman',
      labels = {'x':'Batsman','y':'Total_runs'},)
df_mostruns.head(2)
def glowplots(i,j):
    plt.style.use('cyberpunk')
    print(f' line plot for the follwing feature <{i}> is shown below ↓')
    sn.relplot(x = 'batsman',y = i,color = j,label = i,kind = 'line',height=10,aspect=2.5,data=df_mostruns.head(10))
    mplcyberpunk.add_glow_effects()
    mplcyberpunk.make_lines_glow()
    plt.show()
view_list  =['total_runs','average','strikerate']
color_list = ['yellow','cyan','magenta']
for i,j in zip(view_list,color_list):
    glowplots(i,j)
    print("="*50)
fig1 = px.bar(df_mostruns.head(10),y = 'total_runs',x = 'batsman',
             color = 'total_runs',text = 'total_runs',orientation='v',color_discrete_sequence=px.colors.qualitative.Dark24,)
fig2 = px.bar(df_mostruns.head(10),y = 'average',x = 'batsman',
             color = 'average',text = 'average',orientation='v',color_discrete_sequence=px.colors.qualitative.Dark24)
fig3 = px.bar(df_mostruns.head(10),y = 'strikerate',x = 'batsman',
             color = 'strikerate',text = 'strikerate',orientation='v',color_discrete_sequence=px.colors.qualitative.Dark24)
fig4 = px.bar(df_mostruns.head(10),y = 'numberofballs',x = 'batsman',
             color = 'numberofballs',text = 'numberofballs',orientation='v',color_discrete_sequence=px.colors.qualitative.Dark24)
fig = make_subplots(rows = 2,cols=2,shared_xaxes=False,horizontal_spacing=0.2,vertical_spacing=0.3,
                   subplot_titles=('Total_runs scored by the batsman',
                                  'Average score of a batsman',
                                  'StrikeRate of the batsman',
                                  'Number of balls faced by the batsman'))
fig.add_trace(fig1['data'][0],row = 1,col = 1)
fig.add_trace(fig2['data'][0],row = 1,col = 2)
fig.add_trace(fig3['data'][0],row = 2,col = 1)
fig.add_trace(fig4['data'][0],row = 2,col = 2)
fig.update_layout(height = 700,width = 1000)
fig.show()
def treemapvisual(i,j):
    print(f' tree map analysis of the feature <{i}> of the batsman')
    fig = px.treemap(data_frame=df_mostruns.head(100),path=['batsman'],values=i,
                     color_discrete_sequence=px.colors.qualitative.Plotly_r,
              height=700,title = j)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()
title_list = ['Total_runs scored','number of ball faced','Average score','StrikeRate']
use_list = ['total_runs', 'numberofballs', 'average','strikerate']
for i,j in zip(use_list,title_list):
    treemapvisual(i,j)
    print("="*50)
px.density_heatmap(data_frame=df_mostruns,x = 'batsman',y = 'total_runs',labels = {'x':'batsman','y':'total_runs'},
                  color_continuous_scale=('purple','yellow'))
df_kohli = df_deliveries[(df_deliveries['batsman'] == 'V Kohli')]
df_warner = df_deliveries[(df_deliveries['batsman'] == 'DA Warner')]
df_raina = df_deliveries[(df_deliveries['batsman'] == 'SK Raina')]
df_dhoni = df_deliveries[(df_deliveries['batsman'] == 'MS Dhoni')]
df_gayle = df_deliveries[df_deliveries['batsman'] == 'CH Gayle']
def barsixplot(df):
    fig = px.bar(df[df['batsman_runs'] == 6],x = 'bowler',y = 'batsman_runs',color = 'bowler',opacity=1,
      color_discrete_sequence=px.colors.qualitative.Plotly_r)
    fig.show()
df_top5 = [df_kohli,df_warner,df_raina,df_dhoni,df_gayle]
df_player_name = ['Virat Kohli','David Warner','Suresh Raina','MS dhoni','CH Gayle']
for i,j in zip(df_top5,df_player_name):
    print(f' <{j}> has scored number of sixes with these bowlers')
    barsixplot(i)
    print("="*50)
def foursplot(df):
    fig = px.bar(data_frame=df[df['batsman_runs'] == 4],x = 'bowler',y = 'batsman_runs',color = 'bowler',opacity=1)
    fig.show()
for i,j in zip(df_top5,df_player_name):
    print(f' <{j}> has scored number of fours with these bowlers')
    foursplot(i)
    print("="*50)
def matchmaps(i,j):
    fig = px.treemap(data_frame=df_homeandaway,path=['team'],values=i,color_discrete_sequence=px.colors.qualitative.Dark2_r,
                    title=j)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()
to_eval = ['home_wins', 'away_wins', 'home_matches', 'away_matches',
       'home_win_percentage', 'away_win_percentage']
eval_name = ['home_wins','away_wins','home_matches','away_matches','home_win_percentage','away_win_percentage']
for i,j in zip(to_eval,eval_name):
    print(f' stats for the value <{j}> is shown below ↓')
    matchmaps(i,j)
    print("="*50)
def mplplots(i,j,k):
    print(f' line plot stats for the feature <{k}> is shown below ↓')
    sn.relplot(x = 'team',y = i,data=df_homeandaway,color = j,kind = 'line',
               height=10,
               aspect=2.5
              )
    mplcyberpunk.add_glow_effects()
    mplcyberpunk.make_lines_glow()
    plt.show()
to_eval = ['home_wins', 'away_wins', 'home_matches', 'away_matches',
       'home_win_percentage', 'away_win_percentage']
color_list = ['cyan','yellow','hotpink','red','blue','green']
eval_name = ['home_wins','away_wins','home_matches','away_matches','home_win_percentage','away_win_percentage']
for i,j,k in zip(to_eval,color_list,eval_name):
    #print(f' line plot stats for the feature <{k}> is shown below ↓')
    mplplots(i,j,k)
    print("="*100)
fig1 = px.bar(df_homeandaway,y = 'home_wins',x = 'team',
             color = 'home_wins',text = 'home_wins',orientation='v')
fig2 = px.bar(df_homeandaway,y = 'away_wins',x = 'team',
             color = 'away_wins',text = 'away_wins',orientation='v')
fig3 = px.bar(df_homeandaway,y = 'home_matches',x = 'team',
             color = 'home_matches',text = 'home_matches',orientation='v')
fig4 = px.bar(df_homeandaway,y = 'away_matches',x = 'team',orientation='v',
             color = 'away_matches',text = 'away_matches')
fig5 = px.bar(df_homeandaway,y = 'home_win_percentage',x = 'team',
             orientation='v',color = 'home_win_percentage',text = 'home_win_percentage')
fig6 = px.bar(df_homeandaway,y = 'away_win_percentage',x = 'team',
             orientation='v',color = 'away_win_percentage',text = 'away_win_percentage')
fig = make_subplots(rows = 3,cols = 2,shared_xaxes=False,horizontal_spacing=0.2,vertical_spacing=0.2,
                   subplot_titles=('Stats for the Home Wins ↓',
                                  'Stats for the Away Wins ↓',
                                  'Stats for the Home Matches ↓',
                                  'Stats for the Away Matches ↓',
                                  'Stats for the home Win % ↓',
                                  'Stats for the Away Win % ↓'))
fig.add_trace(fig1['data'][0],row = 1,col = 1)
fig.add_trace(fig2['data'][0],row = 1,col = 2)
fig.add_trace(fig3['data'][0],row = 2,col = 1)
fig.add_trace(fig4['data'][0],row = 2,col = 2)
fig.add_trace(fig5['data'][0],row = 3,col = 1)
fig.add_trace(fig6['data'][0],row = 3,col = 2)
fig.update_layout(height = 1500,width = 1000)
fig.show()
team = df_homeandaway.team
data = [
    {
        'y':df_homeandaway.home_matches,
        'x':df_homeandaway.home_wins,
        'mode':'markers',
        'marker':{
            'color':df_homeandaway.home_matches,
            'size':df_homeandaway.home_wins,
            'showscale':True
        },
        "text":df_homeandaway.team
    }
]
iplot(data)
data = [
    {
        'y':df_homeandaway.away_matches,
        'x':df_homeandaway.away_wins,
        'mode':'markers',
        'marker':{
            'color':df_homeandaway.away_matches,
            'size':df_homeandaway.away_wins,
            'showscale':True
        },
        "text":df_homeandaway.team
    }
]
iplot(data)
toss_winner_cum_winner = df_matches[(df_matches['toss_winner']) == (df_matches['winner'])]
plt.figure(figsize = (35,17.5))
sn.countplot(x = 'winner',data = toss_winner_cum_winner,palette='plasma',saturation=1)

px.bar(data_frame=toss_winner_cum_winner,x = 'winner',color = 'winner',opacity=1)
px.bar(data_frame=toss_winner_cum_winner,x = 'winner',y = 'venue',color = 'venue',opacity=1,
      labels = {'x':'winner_team','y':'Venue'})
df_bowled = df_deliveries[df_deliveries['dismissal_kind'] == 'bowled']
px.bar(data_frame=df_bowled,x = 'bowler',color = 'bowler',opacity=1)
