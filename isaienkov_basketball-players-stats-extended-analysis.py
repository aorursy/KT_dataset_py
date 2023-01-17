import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import plotly.express as px



pd.options.display.max_columns = None
df = pd.read_csv('/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv')

df = df.drop(['height', 'weight'], axis=1)

df.head()
df.columns
missed = pd.DataFrame()

missed['column'] = df.columns

percent = list()



for col in df.columns:

    percent.append(round(100* df[col].isnull().sum() / len(df), 2))

missed['percent'] = percent

missed = missed.sort_values('percent')

missed = missed[missed['percent']>0]



fig = px.bar(

    missed, 

    x='percent', 

    y="column", 

    orientation='h', 

    title='Missed values percent for every column (percent > 0)', 

    height=600, 

    width=600

)



fig.show()
ds = df['League'].value_counts().reset_index()

ds.columns = ['league', 'number of samples']

ds = ds.sort_values(['number of samples'])



fig = px.bar(

    ds, 

    x='number of samples', 

    y="league", 

    orientation='h', 

    title='Leagues presented in dataset', 

    height=1000, 

    width=800

)



fig.show()
ds = df['Season'].value_counts().reset_index()

ds.columns = ['season', 'number of samples']

ds = ds.sort_values(['number of samples'])



fig = px.bar(

    ds, 

    x='number of samples', 

    y="season", 

    orientation='h', 

    title='Seasons presented in dataset', 

    height=600, 

    width=600

)



fig.show()
ds = df['Stage'].value_counts().reset_index()

ds.columns = ['stage', 'number of samples']

ds = ds.sort_values(['number of samples'])



fig = px.pie(

    ds, 

    values='number of samples', 

    names="stage",  

    title='Stages presented in dataset', 

    height=500, 

    width=500

)



fig.show()
fig = px.histogram(

    df, 

    "GP", 

    nbins=90, 

    title='Number of games distribution', 

    width=800, 

    height=600

)



fig.show()
fig = px.histogram(

    df, 

    "MIN", 

    nbins=100, 

    title='Number of minutes distribution', 

    width=800, 

    height=600

)

fig.show()
data = pd.DataFrame()

data['number'] = df['FGM'].copy()

data['legend'] = 'FGM'

data2 = pd.DataFrame()

data2['number'] = df['FGA'].copy()

data2['legend'] = 'FGA'

data = pd.concat([data, data2])
fig = px.histogram(

    data, 

    x="number", 

    nbins=200, 

    color = 'legend',

    title='FGM vs FGA distributions', 

    width=800, 

    height=700

)

fig.show()
data = pd.DataFrame()

data['number'] = df['3PM'].copy()

data['legend'] = '3PM'

data2 = pd.DataFrame()

data2['number'] = df['3PA'].copy()

data2['legend'] = '3PA'

data = pd.concat([data, data2])



fig = px.histogram(

    data, 

    x="number", 

    nbins=80, 

    color = 'legend',

    title='3PM vs 3PA distributions', 

    width=800, 

    height=700

)

fig.show()
data = pd.DataFrame()

data['number'] = df['FTM'].copy()

data['legend'] = 'FTM'

data2 = pd.DataFrame()

data2['number'] = df['FTA'].copy()

data2['legend'] = 'FTA'

data = pd.concat([data, data2])



fig = px.histogram(

    data, 

    x="number", 

    nbins=80, 

    color = 'legend',

    title='FTM vs FTA distributions', 

    width=800, 

    height=700

)

fig.show()
fig = px.histogram(

    df, 

    "TOV", 

    nbins=100, 

    title='Number of turnovers distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "PF", 

    nbins=100, 

    title='Number of personal fouls distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "ORB", 

    nbins=100, 

    title='Number of offensive rebounds distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "DRB", 

    nbins=100, 

    title='Number of defensive rebounds distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "AST", 

    nbins=100, 

    title='Number of assists distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "STL", 

    nbins=100, 

    title='Number of steals distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "BLK", 

    nbins=100, 

    title='Number of blocks distribution', 

    width=800, 

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "PTS", 

    nbins=100, 

    title='Number of points distribution', 

    width=800, 

    height=600

)

fig.show()
ds = df.groupby(['Player', 'nationality'])['Team'].count().reset_index()

ds = ds['nationality'].value_counts().reset_index()

ds.columns = ['nationality', 'number of samples']

ds = ds.sort_values(['number of samples'])

ds = ds.tail(40)

fig = px.bar(

    ds, 

    x='number of samples', 

    y="nationality", 

    orientation='h', 

    title='Top 40 nationalities presented in dataset', 

    height=900, 

    width=900

)

fig.show()