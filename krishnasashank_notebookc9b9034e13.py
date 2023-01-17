project_name = "basketball-player-analysis" # change this
!pip install jovian --upgrade -q
import jovian
jovian.commit(project=project_name)
#Importing all the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
%matplotlib inline
#Reading the dataset
df = pd.read_csv("../input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv")
#Looking at the first 5 entires of the dataset
df.head()
#Looking at the general info of the dataset column-wise
df.info()
#Checking the shape of the dataset
df.shape
#Checking all the columns present in the dataset
df.columns
#Describint the dataset
df.describe().T
#Checking for null values in the dataset
df.isnull().values.any()
#Checking which columns have null values on the dataset
df.isnull().sum()
#Dropping the column "high_school" since it has 30247 null values out of 53798 making it a useless column
df = df.drop(labels=['high_school'], axis = 1)
#Dropping all the rows with null values in any column (we can do this because we have >50000 row values and only about 4500 valued null rows)
df = df.dropna(axis=0)
#Checking the shape of the dataset after removing all the data preprocessing
df.shape
import jovian
jovian.commit(project=project_name)
#Feature Generation for Analysis
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
#Looking at the general info of the dataset column-wise
df.info()
#NBA 2009-2010 Regular Season Players Mean Statistics
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
#NBA 2010-2011 Regular Season Players Mean Statistics
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
#NBA 2009-2010 Regular Season Statistics
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
#NBA 2010-2011 Regular Season Statistics
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
#Top 10 Players Total Matches Played
df_maxgame=df_nba_regs_09_10.sort_values('GP',ascending="False")

df_maxgame=df_maxgame[197:207]

fig = go.Figure(go.Funnel(
    y =  list(df_maxgame['Player']),
    x = list(df_maxgame['GP']))) 
fig.update_layout(title = "Top 10 Players Total Matches Played",title_x=0.5)
fig.show()
#Top 10 Players Season Total Field Goals Made
df_maxpoints=df_nba_regs_09_10.sort_values('FGM',ascending="False")
df_maxpoints=df_maxpoints[197:207]


fig = go.Figure(go.Funnel(
    y =  list(df_maxpoints['Player']),
    x = list(df_maxpoints['FGM']))) 
fig.update_layout(title = "Top 10 Players Season Total Field Goals Made",title_x=0.5)
fig.show()
#Top 10 Players Season Total Assists
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
#Top 10 Players Season Total Rebounds
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
#Top 10 Players  Minutes Played Per Game
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
#Distribution Of Players Points Per Game
fig = go.Figure(data=[go.Histogram(x=df['gameper_pts'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="CadetBlue",
                      xbins=dict(
                      start=0, #start range of bin
                      end=50,  #end range of bin
                      size=5    #size of bin
                      ))])
fig.update_layout(title="Distribution Of Players Points Per Game",xaxis_title="Points",yaxis_title="Counts",title_x=0.5)
fig.show()
#Top 10 Teams And Number Of Players In The Team
df_nba_regs_09_10=df[(df['League']=='NBA')&(df['Stage']=='Regular_Season')&(df['Season']=='2009 - 2010')]

df_team=df_nba_regs_09_10['Team'].value_counts().to_frame().reset_index().rename(columns={'index':'Team','Team':'Count'})

fig = go.Figure([go.Pie(labels=df_team['Team'][0:10], values=df_team['Count'][0:10])])

fig.update_traces(hoverinfo='value+percent', textinfo='label+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title=" Top 10 Teams And Number Of Players In The Team",title_x=0.5)
fig.show()
#Three Points Throws Success Rate & Top 10 Players Three Points Attempts
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
#Number Of Nationality Players
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
import jovian
jovian.commit(project=project_name)
print("Q1. What is the height distributon of the players?")
print("Answer: \n")

b = df[['Player', 'height_cm']]
bb = b.groupby('height_cm').size()
bbb = bb[bb.values > 3000]
fig = plt.figure(figsize=(14, 3), dpi=100)
ax = fig.add_subplot(111)
mean = df['height_cm'].mean()
ax.bar(bb.index, bb.values, width=0.8, color='lightsteelblue')
ax.bar(bbb.index, bbb.values, width=0.8, color='royalblue')
ax.set_xticks(np.arange(165.0, 230.0, 1))
ax.set_xticklabels(np.arange(165.0, 230.0, 1), rotation=90)
ax.set_xlim(164,230)
for x,y in zip(bb.index, bb.values):
    ax.text(x, y+200, y, fontsize=10, rotation=45, horizontalalignment='center')
ax.set_ylim(0,5000)
ax.set_title('Player Height Distribution', fontsize=16, y=1.02)
ax.set_xlabel('Height (CM)', fontsize=12)
ax.set_ylabel('Player Counts', fontsize=12)
ax.yaxis.grid(alpha=0.4, ls='--')
plt.show()
print("Q2. What is the nationality of the top 20 players?")
print("Answer: \n")

c = df[['Player', 'nationality']]
cc = c.sort_values(by=['Player', 'nationality']).reset_index(drop=True)
ccc = cc.groupby('Player', as_index=False).first()
cccc = ccc.groupby('nationality', as_index=False).count().sort_values(by='Player').tail(20)
ccccc = cccc.tail(5)
fig = plt.figure(figsize=(6,8), dpi=100)
ax = fig.add_subplot(111)
ax.barh(cccc['nationality'], cccc['Player'], color='lightsteelblue', height=0.8)
for x,y in zip(cccc['nationality'], cccc['Player']):
    ax.text(y+60, x, y, fontsize=10, horizontalalignment='left')
ax.barh(ccccc['nationality'], ccccc['Player'], color='royalblue', height=0.8)
ax.set_xlim(0,4500)
ax.xaxis.grid(alpha=0.4, ls='--', color='lightsteelblue')
ax.set_title('Player Nationality Distribution', fontsize=16, y=1.02)
ax.set_xlabel('Number of Players', fontsize=12)
ax.set_ylabel('Country of Players', fontsize=12)
plt.show()
print("Q3. What is the Career Score Statistics of the NBA players?")
print("Answer: \n")

e = df.groupby('Player')['PTS'].sum().sort_values(ascending=False)
ee = e.describe()
fig = plt.figure(figsize=(14,4), dpi=100)
ax = fig.add_subplot(111)
ax.boxplot(e.values, widths=0.1, labels=['PTS'], vert=False, sym='+', patch_artist=False, meanline=True, showmeans=True, showcaps=True, showbox=True, showfliers=True)
ax.set_xticks(np.arange(0,30001,2000))
ax.set_xlabel('Total Scores of Player', fontsize=12)
ax.set_title('NBA Players Career Total Score Distribution', fontsize=16, y=1.02)
ax.grid(alpha=0.2)
plt.show()
print("Q4. How has the performance of each player been in NBA?")
print("Answer: \n")

e1 = df['3PM']
e2 = df['FTM']
e3 = df['REB']
e4 = df['AST']
fig = plt.figure(figsize=(4,8), dpi=100)
ax = fig.add_subplot(111)
ax.boxplot([e1.values, e2.values, e3.values, e4.values], widths=0.2, labels=['3PM', 'FTM', 'REB', 'AST'], sym='.', patch_artist=False, vert=True)
ax.set_xlabel('Data Categories', fontsize=12)
ax.set_title('Player data performance distribution in each season', fontsize=14, y=1.02)
ax.set_ylabel('Values', fontsize=12)
ax.set_yticks(np.arange(0,1400,50))
ax.yaxis.grid(alpha=0.2, ls='--')
plt.show()
print("Q5. How has the performance of the Top 20 players w.r.t. Three Pointers, Free Throws and Rebounds?")
print("Answer: \n")

f = df.groupby('Player')['PTS'].sum().sort_values(ascending=False).head(20)
ff = df.loc[df['Player'].isin(f.index)]
fig = plt.figure(figsize=(6,10), dpi=100)
ax = fig.add_subplot(111)
ax.scatter(ff['3PM'], ff['Player'], label='Three Pointer', marker='o', alpha=0.6, color='limegreen')
ax.scatter(ff['FTM'], ff['Player'], label='Free Throw', marker='o', alpha=0.6, color='firebrick')
ax.scatter(ff['REB'], ff['Player'], label='Rebounds', marker='o', alpha=0.6, color='steelblue')
ax.set_xticks(np.arange(0,1001,50))
ax.set_xticklabels(np.arange(0,1001,50), rotation=45)
ax.xaxis.grid(alpha=0.4, ls='--')
ax.set_title('Data Distribution of TOP20 NBA Players', fontsize=16, y=1.01)
ax.set_xlabel('Players\' Performance Per Game', fontsize=12)
ax.set_ylabel('TOP20 Players', fontsize=12)
ax.legend()
plt.show()
import jovian
jovian.commit(project=project_name)
import jovian
jovian.commit(project=project_name)