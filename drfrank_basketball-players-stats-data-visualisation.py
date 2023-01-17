# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
Basketball_Players=pd.read_csv("/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv")
df=Basketball_Players.copy()
df.head()
df.info()
df.shape
df.columns
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df[df.duplicated() == True]
df['gameper_min']=df['MIN']/df['GP']
df['gameper_pts']=df['PTS']/df['GP']
df['gameper_reb']=df['REB']/df['GP']
df['gameper_ast']=df['AST']/df['GP']
df['gameper_blk']=df['BLK']/df['GP']
df['gameper_pf']=df['PF']/df['GP']
df['gameper_tov']=df['TOV']/df['GP']
df['gameper_stl']=df['STL']/df['GP']
df['gameper_3pm']=df['3PM']/df['GP']
df['gameper_2fgm']=df['FGM']/df['GP']
df['gameper_1ftm']=df['FTM']/df['GP']
df['ratio_1ftm']=df['FTM']/df['FTA']
df['ratio_2fgm']=df['FGM']/df['FGA']
df['ratio_3pm']=df['3PM']/df['3PA']
df['age']=2020-df['birth_year']
df.info()
# Multiple Bullet

df_nba_regs_09_10=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2009 - 2010')]
df_nba_regs_09_10=df_nba_regs_09_10.drop(['League','Stage','Season'],axis=1)

avg_game=df_nba_regs_09_10.GP.mean()

avg_min=df_nba_regs_09_10.MIN.mean()

avg_pts=df_nba_regs_09_10.PTS.mean()

avg_ast=df_nba_regs_09_10.AST.mean()

avg_reb=df_nba_regs_09_10.REB.mean()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  avg_game,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Players Mean Game",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_min,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Players Mean Minutes",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,2250]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_pts,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Players Mean Points",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,1000]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_ast,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Players Mean Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,250]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_reb,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Players Mean Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,400]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" NBA 2009-2010 Regular Season Players Mean Statistics ",title_x=0.5)
fig.show()
# Multiple Bullet

df_nba_regs_10_11=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2010 - 2011')]
df_nba_regs_10_11=df_nba_regs_10_11.drop(['League','Stage','Season'],axis=1)
df_nba_regs_10_11


avg_game=df_nba_regs_10_11.GP.mean()

avg_min=df_nba_regs_10_11.MIN.mean()

avg_pts=df_nba_regs_10_11.PTS.mean()

avg_ast=df_nba_regs_10_11.AST.mean()

avg_reb=df_nba_regs_10_11.REB.mean()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  avg_game,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Players Mean Game",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_min,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Players Mean Minutes",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,2250]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_pts,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Players Mean Points",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,1000]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_ast,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Players Mean Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,250]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_reb,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Players Mean Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,400]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" NBA 2010-2011 Regular Season Players Mean Statistics ",title_x=0.5)
fig.show()
# Multiple Bullet

df_nba_regs_09_10=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2009 - 2010')]

df_nba_regs_09_10=df_nba_regs_09_10.drop(['League','Stage','Season'],axis=1)

sum_pts=df_nba_regs_09_10.PTS.sum()

sum_ast=df_nba_regs_09_10.AST.sum()

sum_reb=df_nba_regs_09_10.REB.sum()

sum_blk=df_nba_regs_09_10.BLK.sum()

sum_stl=df_nba_regs_09_10.STL.sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  sum_stl,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Season Sum Steals",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 15000]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_pts,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Season Sum Points",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,200000]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_ast,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Season Sum Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,40000]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_reb,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Season Sum  Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,80000]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_blk,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Season Sum Blocks",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,9000]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" NBA 2009-2010 Regular Season Statistics ",title_x=0.5)
fig.show()
# Multiple Bullet

df_nba_regs_10_11=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2010 - 2011')]
df_nba_regs_10_11=df_nba_regs_10_11.drop(['League','Stage','Season'],axis=1)
df_nba_regs_10_11


sum_pts=df_nba_regs_10_11.PTS.sum()

sum_ast=df_nba_regs_10_11.AST.sum()

sum_reb=df_nba_regs_10_11.REB.sum()

sum_blk=df_nba_regs_10_11.BLK.sum()

sum_stl=df_nba_regs_10_11.STL.sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  sum_stl,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Season Sum Steals",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 15000]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_pts,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Season Sum Points",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,200000]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_ast,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Season Sum Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,40000]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_reb,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Season Sum  Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,80000]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_blk,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Season Sum Blocks",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,9000]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" NBA 2010-2011 Regular Season Statistics ",title_x=0.5)
fig.show()
df_maxgame=df_nba_regs_09_10.sort_values('GP',ascending="False")

df_maxgame=df_maxgame[197:207]

fig = go.Figure(go.Funnel(
    y =  list(df_maxgame['Player']),
    x = list(df_maxgame['GP']))) 
fig.update_layout(title = "Top 10 Players Total Match Played",title_x=0.5)
fig.show()
df_maxpoints=df_nba_regs_09_10.sort_values('FGM',ascending="False")
df_maxpoints=df_maxpoints[197:207]


fig = go.Figure(go.Funnel(
    y =  list(df_maxpoints['Player']),
    x = list(df_maxpoints['FGM']))) 
fig.update_layout(title = "Top 10 Players Season Total Field Goals Made",title_x=0.5)
fig.show()
df_maxassists=df_nba_regs_09_10.sort_values('AST',ascending="False")
df_maxassists=df_maxassists[197:207]


fig = go.Figure(go.Bar(
    x=df_maxassists['Player'],y=df_maxassists['AST'],
    marker={'color': df_maxassists['AST'], 
    'colorscale': 'balance'},  
    text=df_maxassists['AST'],
    textposition = "outside",
))

fig.update_layout(title = "Top 10 Players Season Total Assists",title_x=0.5)
fig.show()
df_maxrebounds=df_nba_regs_09_10.sort_values('REB',ascending="False")

df_maxrebounds=df_maxrebounds[197:207]

fig = go.Figure(go.Bar(
    y=df_maxrebounds['Player'],x=df_maxrebounds['REB'],orientation="h",
    marker={'color': df_maxrebounds['REB'], 
    'colorscale': 'fall'},  
    text=df_maxrebounds['REB'],
    textposition = "outside",
))
fig.update_layout(title = "Top 10 Players Season Total Rebounds",title_x=0.5)
fig.show()

df_game_min=df_nba_regs_09_10.sort_values('gameper_min',ascending="False")
df_game_min=df_game_min[197:207]


fig = go.Figure(go.Bar(
    y=df_game_min['Player'],x=df_game_min['gameper_min'],orientation="h",
    marker={'color': df_game_min['gameper_min'], 
    'colorscale': 'curl'},  
    text=df_game_min['gameper_min'],
    textposition = "outside",
))
fig.update_layout(title = "Top 10 Players  Minutes Played Per Game",title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Histogram(x=df['gameper_pts'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="CadetBlue",
                      xbins=dict(
                      start=0, #start range of bin
                      end=50,  #end range of bin
                      size=5    #size of bin
                      ))])
fig.update_layout(title="Distribution Of Players Points Per Game",xaxis_title="Points",yaxis_title="Counts",title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Histogram(x=df['height_cm'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="DarkSalmon",
                      xbins=dict(
                      start=150, #start range of bin
                      end=280,  #end range of bin
                      size=10    #size of bin
                      ))])
fig.update_layout(title="Distribution Of Players Height [CM]",xaxis_title="Points",yaxis_title="Counts",title_x=0.5)
fig.show()

fig = go.Figure(data=[go.Histogram(x=df['weight'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="LightSkyBlue",
                      xbins=dict(
                      start=100, #start range of bin
                      end=350,  #end range of bin
                      size=10    #size of bin
                      ))])
fig.update_layout(title="Distribution Of Players Weight [Pounds]",xaxis_title="Points",yaxis_title="Counts",title_x=0.5)
fig.show()

df_nba_regs_09_10=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2009 - 2010')]

df_team=df_nba_regs_09_10['Team'].value_counts().to_frame().reset_index().rename(columns={'index':'Team','Team':'Count'})

fig = go.Figure([go.Pie(labels=df_team['Team'][0:10], values=df_team['Count'][0:10])])

fig.update_traces(hoverinfo='value+percent', textinfo='label+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title=" Top 10 Team And Number Of Players In The Team",title_x=0.5)
fig.show()
df_ratio_1ftm=df_nba_regs_09_10.sort_values('ratio_1ftm',ascending="False")
df_ratio_1ftm=df_ratio_1ftm[197:207]
df_ratio_1ftm


fig = go.Figure(go.Bar(
    y=df_ratio_1ftm['Player'],x=df_ratio_1ftm['ratio_1ftm'],orientation="h",
    marker={'color': df_ratio_1ftm['ratio_1ftm'], 
    'colorscale': 'twilight'},  
    text=df_ratio_1ftm['ratio_1ftm'],
    textposition = "outside",
))
fig.update_layout(title = "Top 10 Players Free Throws Success Rate",title_x=0.5)
fig.show()
df_nba_regs_09_10_Rat=df_nba_regs_09_10.copy()
df_nba_regs_09_10_Rat=df_nba_regs_09_10_Rat.dropna(subset=['ratio_3pm'])

df_ratio_3ftm=df_nba_regs_09_10_Rat.sort_values('ratio_3pm',ascending="False")
df_ratio_3ftm=df_ratio_3ftm[189:199]

fig = make_subplots(rows=2, cols=1,
                   subplot_titles=("Top 10 Players Three Points Success Rate",
                                   "Top 10 Players Three Points Attempts",))  # Subplot titles

fig.add_trace(go.Bar(
    y=df_ratio_3ftm['Player'],x=df_ratio_3ftm['ratio_3pm'],orientation="h",
    marker={'color': df_ratio_3ftm['ratio_3pm'], 
    'colorscale': 'ylorrd'},  
    text=df_ratio_3ftm['ratio_3pm'],
    textposition = "outside"),
    row=1, col=1         
)
fig.add_trace(go.Bar(
    y=df_ratio_3ftm['Player'],x=df_ratio_3ftm['3PA'],orientation="h",
    marker={'color': df_ratio_3ftm['3PA'], 
    'colorscale': 'twilight'},  
    text=df_ratio_3ftm['3PA'],
    textposition = "outside"),
    row=2, col=1           
)
fig.update_layout(height=800, width=600,title = "Three Points Throws",title_x=0.5)
fig.show()

df_game_play=df_nba_regs_09_10.sort_values('PTS',ascending="False")
df_game_play=df_game_play[197:207]
df_game_play


fig = make_subplots(rows=2, cols=1,
                   subplot_titles=("Top 10 Players Total Points ",
                                   "Top 10 Total Points Players Games Played ",))  # Subplot titles

fig.add_trace(go.Bar(
    y=df_game_play['Player'],x=df_game_play['PTS'],orientation="h",
    marker={'color': df_game_play['PTS'], 
    'colorscale': 'ylorrd'},  
    text=df_game_play['PTS'],
    textposition = "outside"),
    row=1, col=1         
)
fig.add_trace(go.Bar(
    y=df_game_play['Player'],x=df_game_play['GP'],orientation="h",
    marker={'color': df_game_play['GP'], 
    'colorscale': 'twilight'},  
    text=df_game_play['GP'],
    textposition = "outside"),
    row=2, col=1           
)
fig.update_layout(height=800, width=600,title = "Top 10 Players Total Points And  Games Played ",title_x=0.5)
fig.show()
df_nba_regs_15_16=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2015 - 2016')]
df_nba_regs_15_16_Leb_JA=df_nba_regs_15_16[df_nba_regs_15_16['Player']=='LeBron James']
#df_nba_regs_15_16_Leb_JA


sum_pts=df_nba_regs_15_16_Leb_JA.PTS.sum()

sum_ast=df_nba_regs_15_16_Leb_JA.AST.sum()

sum_reb=df_nba_regs_15_16_Leb_JA.REB.sum()

sum_blk=df_nba_regs_15_16_Leb_JA.BLK.sum()

sum_game=df_nba_regs_15_16_Leb_JA.GP.sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  sum_game,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Season Games Played",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_pts,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Season Sum Points",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,2000]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_ast,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Season Sum Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,800]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_reb,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Season Sum  Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,800]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_blk,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Season Sum Blocks",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,100]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" LeBron James  ",title_x=0.5)
fig.show()
df_nba_playof_15_16=df[(df['League']=='NBA')&(df['Stage']=='Playoffs')&(df['Season']=='2015 - 2016')]
df_nba_playof_15_16_Leb_JA=df_nba_playof_15_16[df_nba_playof_15_16['Player']=='LeBron James'] 


sum_pts=df_nba_playof_15_16_Leb_JA.PTS.sum()

sum_ast=df_nba_playof_15_16_Leb_JA.AST.sum()

sum_reb=df_nba_playof_15_16_Leb_JA.REB.sum()

sum_blk=df_nba_playof_15_16_Leb_JA.BLK.sum()

sum_game=df_nba_playof_15_16_Leb_JA.GP.sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  sum_game,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Playoff Games Played",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 50]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_pts,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Playoff Sum Points",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,750]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_ast,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Playoff Sum Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,200]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_reb,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Playoff Sum  Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,300]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_blk,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Playoff Sum Blocks",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" LeBron James  ",title_x=0.5)
fig.show()
df_nba_playof_15_16=df[(df['League']=='NBA')&(df['Stage']=='Playoffs')&(df['Season']=='2015 - 2016')]
df_nba_playof_15_16_Leb_JA=df_nba_playof_15_16[df_nba_playof_15_16['Player']=='LeBron James'] 
#df_nba_playof_15_16_Leb_JA


sum_pts=df_nba_playof_15_16_Leb_JA.gameper_pts.sum()

sum_ast=df_nba_playof_15_16_Leb_JA.gameper_ast.sum()

sum_reb=df_nba_playof_15_16_Leb_JA.gameper_reb.sum()

sum_blk=df_nba_playof_15_16_Leb_JA.gameper_blk.sum()

sum_game_min=df_nba_playof_15_16_Leb_JA.gameper_min.sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  sum_game_min,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Minutes Played ",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 50]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_pts,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Points",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_ast,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :" Assists",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_reb,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :" Rebounds",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,20]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = sum_blk,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Blocks",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,5]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" LeBron James  ",title_x=0.5)
fig.show()
df_Euro_16_17=df[(df['League']=='Euroleague')&(df['Stage']=='International')&(df['Season']=='2016 - 2017')]

df_nati=df_Euro_16_17.nationality.value_counts().to_frame().reset_index().rename(columns={'index':'nationality','nationality':'Count'})[0:12]
df_nati=df_nati.sort_values('Count',ascending="False")


fig = go.Figure(go.Bar(
    y=df_nati['nationality'],x=df_nati['Count'],orientation="h",
    marker={'color': df_nati['Count'], 
    'colorscale': 'curl'},  
    text=df_nati['Count'],
    textposition = "outside",
))
fig.update_layout(title = "Number Of Nationality Players ",title_x=0.5)
fig.show()

df_Euro_16_17=df[(df['League']=='Euroleague')&(df['Stage']=='International')&(df['Season']=='2016 - 2017')]
df_Euro_16_17
fig = go.Figure()
fig.add_trace(go.Box(y=df_Euro_16_17['age'],
                     boxmean='sd', # mean and SD visible on plot
                     marker_color="darkorchid",
                     name="Age"))
fig.update_layout(title="Distribution of Age",title_x=0.5)
fig.show()
