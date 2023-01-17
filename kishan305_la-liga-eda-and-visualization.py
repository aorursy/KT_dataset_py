import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.figure_factory as ff
data = pd.read_csv('/kaggle/input/la-liga-results-19952020/LaLiga_Matches_1995-2020.csv')
display(data.head())

display(data.tail())
data.describe()
data.info()
data.isnull().values.any()
data[data.isnull().any(axis=1)]
data.loc[136,'HTHG'] = 0

data.loc[1472,'HTHG'] = 0



data.loc[136,'HTAG'] = 0

data.loc[1472,'HTAG'] = 0



data.loc[136,'HTR'] = 'D'

data.loc[1472,'HTR'] = 'D'
data.isnull().values.any()
# Get total match in each season



plt.rcParams['figure.figsize'] = (20,10)

plt.style.use('dark_background')



sns.countplot(data['Season'], palette = 'gnuplot')



plt.tick_params(labelsize=15) 

plt.title('Number of Matches Played in each Season', fontweight = 30, fontsize =20)

plt.xticks(rotation = 90)

plt.show()

plt.style.use('ggplot')



plt.figure(figsize=(20,10))

fig = plt.subplots()



plt.tick_params(labelsize=15) 



plt.subplot(2,2,1)

plt.boxplot(data['FTHG'])

plt.ylabel('Goals',fontsize=14)

plt.title('Distribution of FULL TIME HOME GOALS', fontsize=20, color='magenta')



plt.subplot(2,2,2)

plt.boxplot(data['FTAG'])

plt.ylabel('Goals',fontsize=14)

plt.title('Distribution of FULL TIME AWAY GOALS', fontsize=20, color='magenta')



plt.subplot(2,2,3)

plt.boxplot(data['HTHG'])

plt.ylabel('Goals',fontsize=14)

plt.title('Distribution of HALF TIME HOME GOALS', fontsize=20, color='magenta')



plt.subplot(2,2,4)

plt.boxplot(data['HTAG'])

plt.ylabel('Goals',fontsize=14)

plt.title('Distribution of HALF TIME AWAY GOALS', fontsize=20, color='magenta')



plt.show()
# Get total matches played as home team by each team

team_home = data['HomeTeam'].value_counts()



team_home = pd.DataFrame(team_home)

team_home['Matches Played as Home Team'] = team_home['HomeTeam']

team_home['HomeTeam'] = team_home.index



team_home.reset_index(drop=True, inplace=True)
# Selecting only top 20 teams for clean Visualization

team_home = team_home.head(20)



sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

plt.figure(figsize=(20,10))

fig = sns.barplot(x='HomeTeam',

                   y='Matches Played as Home Team',

                   data= team_home)

fig.set_xticklabels(fig.get_xticklabels(), rotation=90)

temp = fig.set_title('Number of matches by each Club as Home Team', fontsize=20)
team_away = data['AwayTeam'].value_counts()



team_away = pd.DataFrame(team_away)

team_away['Matches Played as Away Team'] = team_away['AwayTeam']

team_away['AwayTeam'] = team_away.index



team_away.reset_index(drop=True, inplace=True)
# Selecting only top 20 teams for clean Visualization

team_away = team_away.head(20)



sns.set_style("darkgrid")

plt.figure(figsize=(20,10))

fig = sns.barplot(x='AwayTeam',

                   y='Matches Played as Away Team',

                   data= team_away,

                   palette = 'cubehelix')

fig.set_xticklabels(fig.get_xticklabels(), rotation=90)

temp = fig.set_title('Number of matches Played by each Club as Away Team', fontsize=20)
total_match = team_home['Matches Played as Home Team'] + team_away['Matches Played as Away Team']

x = team_home['HomeTeam'].tolist()

y = total_match.tolist()

team_match = pd.DataFrame(list(zip(x, y)), 

               columns =['Team', 'Matches Played'])
# Selecting only top 20 teams for clean Visualization

team_match = team_match.head(20)



sns.set_style("darkgrid")

plt.figure(figsize=(20,10))

fig = sns.barplot(x='Team',

                   y='Matches Played',

                   data= team_match, palette='cool')

fig.set_xticklabels(fig.get_xticklabels(), rotation=90)

temp = fig.set_title('Number of matches by each Club', fontsize=20)
data1 = data['Date'].value_counts()



data1 = pd.DataFrame(data1)

data1['Number of Matches Played'] = data1.Date

data1['Date'] = data1.index



data1.reset_index(drop=True, inplace=True)



data1.max()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['FTHG'], palette = 'hsv')

plt.title('Distribution of Home Team Goals', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['FTAG'], palette = 'copper')

plt.title('Distribution of Away Team Goals', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(data.corr(), cmap = 'Wistia', annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.show()
data['FTHG'].sum()
data['FTAG'].sum()
data['FTHG'].sum() + data['FTAG'].sum()
ftr = data['FTR'].value_counts()



ftr = pd.DataFrame(ftr)

ftr['COUNT'] = ftr.FTR

ftr['FTR'] = ftr.index



ftr.reset_index(drop=True, inplace=True)
import plotly.express as px

fig = px.pie(ftr, values='COUNT', names='FTR', title='Results of LaLiga')

fig.show()
sns.set_style("ticks")



plt.figure(figsize=(15,5))

fig = sns.barplot(x='FTR',

                   y='COUNT',

                   data= ftr,

                   palette = 'rocket')

fig.set_xticklabels(fig.get_xticklabels(), rotation=0)

temp = fig.set_title('Result at Full Time', fontsize=20)
# Get total wins for each teams 



team_list = list(data['HomeTeam'].unique())

team_win_dic = {}



for team in team_list:

    win_cnt = 0

    

    for i in range(len(data)):

        if data['FTR'][i] == 'H' and data['HomeTeam'][i] == team:

            win_cnt += 1

        elif data['FTR'][i] == 'A' and data['AwayTeam'][i] == team:

            win_cnt += 1

               

    team_win_dic[team] = win_cnt

    



teams_win = sorted(team_win_dic.items(), key=lambda team : team[1], reverse=True)

team_win = pd.DataFrame(teams_win, columns=['Team','Number of Wins']).head(20)

# Selecting only top 20 teams for clean Visualization
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=team_win['Team'],

                               y=team_win['Number of Wins'],

                               mode = 'lines + markers'))

fig.update_layout(title = 'Wins for each Team')

fig.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['HTHG'], palette = 'hsv')

plt.title('Distribution of Home Team Goals', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['HTAG'], palette = 'rainbow')

plt.title('Distribution of Away Team Goals', fontsize = 20)

plt.show()
int(data['HTHG'].sum())
int(data['HTAG'].sum())
int(data['HTHG'].sum() + data['HTAG'].sum())
htr = data['HTR'].value_counts()

htr = pd.DataFrame(htr)

htr['COUNT'] = htr.HTR

htr['HTR'] = htr.index

htr.reset_index(drop=True, inplace=True)
import plotly.express as px

fig = px.pie(htr, values='COUNT', names='HTR', title='Results of LaLiga at Half Time')

fig.show()
plt.figure(figsize=(15,5))

fig = sns.barplot(x='HTR',

                   y='COUNT',

                   data= htr ,

                   palette = 'RdYlBu')

fig.set_xticklabels(fig.get_xticklabels(), rotation=0)

temp = fig.set_title('Result at Half Time', fontsize=20)
# Get total Half Time Wins for each teams 



team_list = list(data['HomeTeam'].unique())

halftime_win_dic = {}



for team in team_list:

    win_cnt = 0

    

    for i in range(len(data)):

        if data['HTR'][i] == 'H' and data['HomeTeam'][i] == team:

            win_cnt += 1

        elif data['HTR'][i] == 'A' and data['AwayTeam'][i] == team:

            win_cnt += 1

               

    halftime_win_dic[team] = win_cnt

    



halftime_win = sorted(halftime_win_dic.items(), key=lambda team : team[1], reverse=True)

halftime_win = pd.DataFrame(halftime_win, columns=['Team','Number of Wins']).head(20)

# Selecting only top 20 teams for clean Visualization
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=halftime_win['Team'],

                               y=halftime_win['Number of Wins'],

                               mode = 'lines + markers',

                               ))

fig.update_layout(title = 'Wins for each Team if LaLiga lasted for 45 minutes')

fig.show()
# Get total Losses for each teams 



team_list = list(data['HomeTeam'].unique())

team_loss_dic = {}



for team in team_list:

    loss_cnt = 0

    

    for i in range(len(data)):

        if data['FTR'][i] == 'A' and data['HomeTeam'][i] == team:

            loss_cnt += 1

        elif data['FTR'][i] == 'H' and data['AwayTeam'][i] == team:

            loss_cnt += 1

               

    team_loss_dic[team] = loss_cnt

    



teams_loss = sorted(team_loss_dic.items(), key=lambda team : team[1], reverse=True)
team_loss = pd.DataFrame(teams_loss, columns=['Team','Number of Losses']).head(20)

# Selecting only top 20 teams for clean Visualization



sns.set_style("whitegrid")

plt.figure(figsize=(20,10))

fig = sns.barplot(x='Team',

                   y='Number of Losses',

                   data= team_loss, 

                   palette='cool')

fig.set_xticklabels(fig.get_xticklabels(), rotation=90)

temp = fig.set_title('Number of matches Lost by each Club', fontsize=20)
team_loss = pd.DataFrame(teams_loss, columns=['Team','Number of Losses']).tail(20)

# Selecting only top 20 teams for clean Visualization



sns.set_style("whitegrid")

plt.figure(figsize=(20,10))

fig = sns.barplot(x='Team',

                   y='Number of Losses',

                   data= team_loss, 

                   palette='copper')

fig.set_xticklabels(fig.get_xticklabels(), rotation=90)

temp = fig.set_title('Number of matches Lost by each Club', fontsize=20)
# Get total Draws for each teams 



team_list = list(data['HomeTeam'].unique())

team_draw_dic = {}



for team in team_list:

    draw_cnt = 0

    

    for i in range(len(data)):

        if data['FTR'][i] == 'D' and data['HomeTeam'][i] == team:

            draw_cnt += 1

        elif data['FTR'][i] == 'D' and data['AwayTeam'][i] == team:

            draw_cnt += 1

               

    team_draw_dic[team] = draw_cnt

    



teams_draw = sorted(team_draw_dic.items(), key=lambda team : team[1], reverse=True)

team_draw = pd.DataFrame(teams_draw, columns=['Team','Number of Draws']).head(20)

# Selecting only top 20 teams for clean Visualization
sns.set_style("whitegrid")

plt.figure(figsize=(20,10))

fig = sns.barplot(x='Team',

                   y='Number of Draws',

                   data= team_draw, 

                   palette='rocket')

fig.set_xticklabels(fig.get_xticklabels(), rotation=90)

temp = fig.set_title('Number of matches Drawn by each Club', fontsize=20)