import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go



!pip install calplot

!pip install calmap

import calmap

import calplot

import matplotlib.pyplot as plt

from IPython.display import clear_output



%matplotlib inline

clear_output()







games_df = pd.read_csv('../input/nfl-big-data-bowl-2021/games.csv')

players_df = pd.read_csv('../input/nfl-big-data-bowl-2021/players.csv')

plays_df = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')

teams_df = pd.read_csv('../input/nfl-team-names/teams.csv')

positions_df = pd.read_csv('../input/nfl-team-names/positions.csv')
df = games_df['gameDate'].value_counts().reset_index().copy()

df.columns = ['date', 'games']

df.sort_values(['games'], inplace = True, ascending = False)



fig = px.bar(df,

             x='date',

             y="games", 

             color = "games",

             title='Number of Games on each Date',

             height=400,

             width=800,

             color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title_x=0.5, xaxis_title = 'Dates', yaxis_title = '#games' )

fig.update_xaxes(type='category', tickangle = 60 )

fig.show()
game_date = pd.to_datetime(games_df['gameDate']).value_counts()

# game_date = pd.Series(game_date.values, index= game_date.index)

plt.figure()

calplot.calplot(game_date, cmap='YlGn', figsize=(15.,3.5))

plt.title('Game Played Over time')

clear_output()
df = games_df['gameTimeEastern'].value_counts().reset_index()

df.columns = ['time', 'games']

df.sort_values(['games'], ascending = True, inplace = True)



fig = px.bar(

    df, 

    y='time', 

    x="games", 

    orientation='h', 

    color = "games",

    title='Games Played in Time of Day', 

    height=400, 

    width=800,

    color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title_x = 0.5, xaxis_title = '#games')

fig.show()
fig = go.Figure(data=[go.Pie(labels=df['time'], values=df['games'], hole=.5)])

fig.update_layout(title = 'Game Play Time Propotions', title_x = 0.5)

fig.show()
df = games_df.groupby(['week']).count()['gameId'].copy()

df = pd.DataFrame({'week': df.index, 'games': df.values})

fig = px.bar(df, x='week', y = 'games', color= 'games', color_continuous_scale=px.colors.sequential.Viridis_r)

fig.update_xaxes(type='category')

fig.update_layout(title = 'Number of Gemes Played Weekly', title_x = 0.5, xaxis_title = 'Weeks ->', yaxis_title = '#games')

fig.show()
df = games_df['homeTeamAbbr'].value_counts().copy()

df = pd.DataFrame({'homeTeamAbbr': df.index, 'games': df.values })

df.sort_values(['homeTeamAbbr'], inplace = True)

df.reset_index(inplace = True, drop = True)



team_names = teams_df.set_index('short').to_dict()['long']





fig = px.bar(df.replace(team_names), 

             x='homeTeamAbbr',

             y= 'games',

             color = 'games',

             title='Home Games for Each Team',

#              height=400,

#              width=800,

             color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title_x=0.5, xaxis_title = 'Home Teams', yaxis_title = '#games' )

fig.update_xaxes(tickangle = 60)

fig.show()

# Height conversion function

def height_convert(x):

    if len(x)>2:

        [ft, inch] = x.split('-')

        return int(ft)*12 + int(inch)

    else:

        return int(x)



players_df['height'] = players_df.apply(lambda x: height_convert(x.height), 1)

# Convert all the dates to one single format

players_df['birthDate'] = pd.to_datetime(players_df.birthDate)

players_df.head(5)
df = players_df['position'].value_counts()

df = pd.DataFrame({'position': df.index, 'count': df.values})

df.sort_values(['count'], inplace = True, ascending = True)

positions_map = positions_df.set_index('short').to_dict()['long']



fig = px.bar(df.replace(positions_map),

             y='position', 

             x="count",

             color = 'count',

             orientation='h', 

             title='Number of Players at Different Positions',

             height=600,

             width=800)

fig.update_layout(title_x = 0.5, xaxis_title = 'Position', yaxis_title = 'Player Counts')

fig.show()
df = players_df['collegeName'].value_counts().reset_index().copy()

df.columns = ['college', 'players']

df.sort_values('players', ascending= True, inplace = True)



fig = px.bar(df.tail(30), 

    y='college', 

    x="players", 

    orientation='h',

    color = "players",

    title='Top 30 colleges by number of players',

    height=900,

    width=800

)

fig.update_layout(title_x = 0.5, xaxis_title = 'Player Count', yaxis_title = 'College Name')

fig.show()
fig = px.box(players_df.replace(positions_map) , y="height", color="position", title="Height Distribution by Player Position", width = 1000)

fig.update_layout(title_x = 0.5, xaxis_title = 'Player Position', yaxis_title = 'Height Distrubution')

fig.show()
df = players_df.groupby(['position']).mean()['weight'].copy()

df = pd.DataFrame({'position': df.index, 'weight': df.values})

df.sort_values(['weight'], inplace = True, ascending = False)



positions_map = positions_df.set_index('short').to_dict()['long']





fig = px.bar(df.replace(positions_map), 

             x='position',

             y= 'weight',

             color = 'weight',

             title='Average Weight of The Players by Position',

#              height=400,

             width=800,

             color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title_x=0.5, xaxis_title = 'Player Position', yaxis_title = 'Average Weight' )

fig.update_xaxes(tickangle = 60)

# fig.update_yaxes(type='category')

fig.show()
fig = px.box(players_df.replace(positions_map) , y="weight", color="position", title="Weight Distribution by Player Position", width = 1000)

fig.update_layout(title_x = 0.5, xaxis_title = 'Player Position', yaxis_title = 'Weight Distrubution')

fig.show()
df = plays_df['possessionTeam'].value_counts().reset_index()

df.columns = ['team', 'plays']

df = df.sort_values('plays')



fig = px.bar(df.replace(team_names), 

             y='team', 

             x="plays", 

             orientation='h',

             color = "plays",

             title='Number of Plays for Each Team',

             height=800,

             width=800,

             color_continuous_scale=px.colors.sequential.Viridis

)

fig.update_layout(title_x = 0.5, xaxis_title = "No of Plays", yaxis_title = "Teams")

fig.show()
df = plays_df['playType'].value_counts().reset_index()

df.columns = ['type', 'plays']

df.sort_values(['plays'], inplace = True, ascending = False)



fig = px.pie(

    df, 

    names='type', 

    values="plays",  

    title='Number of plays of every type',

    height=600,

    width=600, 

    hole = 0.5

    

)

fig.update_layout(title_x = 0.5)

fig.show()
df = plays_df['yardlineNumber'].value_counts().reset_index()

df.columns = ['yardline', 'plays']

df.sort_values('plays', inplace = True)



fig = px.bar(df, 

    x='yardline', 

    y="plays",  

    color = "plays", 

    title='Number of Plays for Every Yard Line',

    height=600,

    width=800

)

fig.update_layout(title_x = 0.5, xaxis_title = 'Yard Line Number', yaxis_title = 'Number of Plays')

fig.show()
df = plays_df['offenseFormation'].value_counts().reset_index()

df.columns = ['offenseFormation', 'plays']

df = df.sort_values('plays')



fig = px.pie(df, 

             names='offenseFormation',

             values="plays",  

             title='Number of Plays for Every Offense Formation Type',

             height=600,

             width=600, 

             hole = 0.4)



fig.update_layout(title_x = 0.5)

fig.show()