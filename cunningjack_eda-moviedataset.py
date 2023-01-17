import numpy as np 
import pandas as pd 

import os

files_wanted = ['movies_metadata.csv']
file_paths = list()

for dirname, _, filenames in os.walk('/kaggle/input'):
     for filename in filenames:
            if filename in files_wanted:
                file_paths.append(str(dirname + "/" + filename))
meta_df = pd.read_csv(file_paths[0], low_memory=False)
meta_df.head()
meta_df.shape
meta_df.drop(['belongs_to_collection', 'homepage', 'tagline', 'poster_path', 'overview', 'imdb_id', 'spoken_languages'], inplace=True, axis=1)

column_changes = ['production_companies', 'production_countries', 'genres']

json_shrinker_dict = dict({'production_companies': list(), 'production_countries': list(), 'genres': list()})

meta_df.dropna(inplace=True)
meta_df.isnull().sum(axis=0)
import ast
for col in column_changes:
    if col == 'production_companies':
        for i in meta_df[col]:
            i = ast.literal_eval(i)
            if len(i) < 1:
                json_shrinker_dict['production_companies'].append(None)

            for element in i:
                json_shrinker_dict['production_companies'].append(element['name'])
                break
    elif col == 'production_countries':
        for i in meta_df[col]:
            i = ast.literal_eval(i)
            if len(i) < 1:
                json_shrinker_dict['production_countries'].append(None)
            for element in i:
                json_shrinker_dict['production_countries'].append(element['iso_3166_1'])
                break
    else:
        for i in meta_df[col]:
            i = ast.literal_eval(i)
            if len(i) < 1:
                json_shrinker_dict['genres'].append(None)

            for element in i:
                json_shrinker_dict['genres'].append(element['name'])
                break

for i in column_changes:
    meta_df[i] = json_shrinker_dict[i]

meta_df.dropna(inplace=True)

meta_df['budget'] = meta_df['budget'].astype(int)
meta_df.head()
new_df = meta_df.loc[(meta_df['adult']=='False') & (meta_df['status']=='Released') & (meta_df['video']==False)]
new_df['year'] = pd.to_datetime(new_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
new_df.drop(['adult','status','video','release_date','id','original_title'], inplace=True, axis=1)
new_df.head()
vote_counts = new_df['vote_count'].astype('int')
vote_averages = new_df['vote_average'].astype('int')
C = vote_averages.mean()
C
m = vote_counts.quantile(0.90)
m
qualified = new_df[(new_df['vote_count'] >= m)]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False)
qualified.shape
qualified[['title','original_language','genres','popularity','wr']].head(10)
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
fig = px.scatter(qualified, x='wr', y='popularity', hover_data=['title'], color='genres', width=800, height=800)
fig.update_layout(
    title='The Relationship between Ratings and Popularity',
    xaxis_title='Weighted Rating',
    yaxis_title='Popularity',
    font=dict(
        size=16
    )
)
iplot(fig)
genre_ratings_df = qualified.groupby(['original_language'])['wr'].count()

fig = go.Figure([
    go.Bar(
        x=genre_ratings_df.index,
        y=genre_ratings_df.values,
        text=genre_ratings_df.values.round(1),
        textposition='auto',
        marker_color=['#94447f',
                      '#5796ef',
                      '#8a59c0',
                      '#288abf',
                      '#0ab78d',
                      '#4ed993',
                      '#7d3970',
                      '#b3dc67',
                      '#dc560a',
                      '#0079fe',
                      '#98d3a8',
                      '#d5105a',
                      '#d04dcf',
                      '#58c7a2',
                      '#7bf1f8',
                      '#244155',
                      '#587b77',
                      '#c64ac2',
                      '#5e805d',
                      '#ebab95']
    )])

fig.update_layout(
    title='Count of Movies under each language',
    xaxis_title='Language',
    yaxis_title='Count',
    width=800,
    height=800,
    font=dict(
        size=16
    )
)

fig.layout.template = 'seaborn'

iplot(fig)
fig = px.box(qualified, y='wr', x='original_language', hover_data=['title'], color='original_language', width=800, height=800)
fig.update_layout(
    title='Distribution of Ratings for each language',
    yaxis_title='Language',
    xaxis_title='Rating',
    font=dict(
        size=16
    )
)

iplot(fig)
genre_ratings_df = qualified.groupby(['genres'])['wr'].count()

fig = go.Figure([
    go.Bar(
        x=genre_ratings_df.index,
        y=genre_ratings_df.values,
        text=genre_ratings_df.values.round(1),
        textposition='auto',
        marker_color=['#94447f',
                      '#5796ef',
                      '#8a59c0',
                      '#288abf',
                      '#0ab78d',
                      '#4ed993',
                      '#7d3970',
                      '#b3dc67',
                      '#dc560a',
                      '#0079fe',
                      '#98d3a8',
                      '#d5105a',
                      '#d04dcf',
                      '#58c7a2',
                      '#7bf1f8',
                      '#244155',
                      '#587b77',
                      '#c64ac2',
                      '#5e805d',
                      '#ebab95']
    )])

fig.update_layout(
    title='Count of Movies in each Genre',
    xaxis_title='Genre',
    yaxis_title='Count of Movies',
    width=800,
    height=1000,
    font=dict(
        size=16
    )
)

fig.layout.template = 'seaborn'

iplot(fig)
fig = px.box(qualified, y='wr', x='genres', hover_data=['title'], color='genres', width=800, height=800)
fig.update_layout(
    title='The Relationship between Genre and Ratings',
    yaxis_title='Genre',
    xaxis_title='Rating',
    font=dict(
        size=16
    )
)

iplot(fig)
fig = px.scatter(qualified, y='runtime', x='wr', hover_data=['title'], color='genres', width=800, height=800)
fig.update_layout(
    title='The Relationship between Runtime and Rating',
    yaxis_title='Runtime',
    xaxis_title='Rating',
    font=dict(
        size=16
    )
)

iplot(fig)