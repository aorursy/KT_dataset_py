import numpy as np

import pandas as pd

        

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go



pd.set_option('display.max_columns', None)
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/games.csv')
games
games_num = games['gameDate'].value_counts().reset_index()

games_num.columns = ['Date' , "Games"]

games_num = games_num.sort_values('Games' , ascending = True)



fig = px.bar(

      games_num,

    y = 'Date',

    x = 'Games',

    orientation = 'h',

    color = 'Games',

    title = 'Number of games for every Date',

    height = 500,

    width = 500





)



fig.show()
check = games['gameTimeEastern'].value_counts().reset_index()

check.columns = ['Time' , 'Games']

check = check.sort_values('Games')



fig = px.bar(

    check,

    x = 'Games',

    y = 'Time',

    color = 'Games',

    orientation = 'h',

    title = 'Number of games for every Time',

    height = 500,

    width = 500

    





)



fig.show()
check = games['homeTeamAbbr'].value_counts().reset_index()

check.columns = ['Team', 'Games']

check = check.sort_values('Games')



fig = px.bar(

    check, 

    y='Team', 

    x="Games", 

    orientation='h',

    color = 'Games',

    title='Number of games for every team (home)', 

    height=500, 

    width=500

)



fig.show()
check = games['visitorTeamAbbr'].value_counts().reset_index()

check.columns = ['Team', 'Games']

check = check.sort_values('Games')



fig = px.bar(

    check, 

    y='Team', 

    x="Games", 

    orientation='h', 

    title='Number of games for every team (Visitor)', 

    height=500, 

    width=500

)



fig.show()
check = games['week'].value_counts().reset_index()

check.columns = ['Week_Numeric', 'Games']

check = check.sort_values('Games')



fig = px.bar(

    check, 

    y='Week_Numeric', 

    x="Games", 

    orientation='h',

    color = 'Games',

    title='Number of games for every week', 

    height=500, 

    width=500

)



fig.show()
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/players.csv')

players
players['height'] = players['height'].str.replace('-', '.')

players['height'] = players['height'].astype(np.float32)

players.loc[players['height']>10, 'height'] /= 12

players
fig = px.histogram(

    players, 

    x="height",

    width=500,

    height=500,

    nbins=20,

        title='Players height distribution'

)



fig.show()
fig = px.histogram(

    players, 

    x="weight",

    width=500,

    height=500,

    nbins=20,

        title='Players weight distribution'

)



fig.show()
check = players.collegeName.value_counts().reset_index()

check.columns = ['collegeName' , 'Players']

check.sort_values('Players' , inplace=True)



fig = px.bar(

   check.tail(20),

    x = 'Players',

    y = 'collegeName',

    orientation = 'h',

    title = 'Top 20 colleges by number of Players',

    color = 'Players',

    height = 900,

    width = 800





)



fig.show()
check = players.position.value_counts().reset_index()

check.columns = ['Position' , 'Players']

check.sort_values('Players' , inplace=True)



fig = px.bar(

   check,

    x = 'Players',

    y = 'Position',

    orientation = 'h',

    title = 'Top positions by number of players',

    color = 'Players',

    height = 900,

    width = 800





)



fig.show()
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/plays.csv')

plays
check = plays['possessionTeam'].value_counts().reset_index()

check.columns = ['team', 'plays']

check = check.sort_values('plays')



fig = px.bar(

    check, 

    y='team', 

    x="plays", 

    orientation='h', 

    color = 'plays',

    title='Number of plays for every team',

    height=800,

    width=800

)



fig.show()
check = plays['playType'].value_counts().reset_index()

check.columns = ['type', 'plays']

check = check.sort_values('plays')



fig = px.pie(

    check, 

    names='type', 

    values="plays",  

    title='Number of plays of every type',

    height=600,

    width=600

)



fig.show()
check = plays['yardlineNumber'].value_counts().reset_index()

check.columns = ['yardline', 'plays']

check = check.sort_values('plays')



fig = px.bar(

    check, 

    x='yardline', 

    y="plays",  

    title='Number of plays for every yardline',

    height=600,

    width=800,

    color = 'plays'

)



fig.show()
check = plays['offenseFormation'].value_counts().reset_index()

check.columns = ['offenseFormation', 'plays']

check = check.sort_values('plays')



fig = px.pie(

    check, 

    names='offenseFormation', 

    values="plays",  

    title='Number of plays for every offense formation type',

    height=600,

    width=600

)



fig.show()
check = plays['defendersInTheBox'].value_counts().reset_index()

check.columns = ['defendersInTheBox', 'plays']

check = check.sort_values('plays')



fig = px.bar(

    check, 

    x='defendersInTheBox', 

    y="plays",  

    title='Number of plays for every number of defenders in the box',

    height=600,

    width=800

)



fig.show()


