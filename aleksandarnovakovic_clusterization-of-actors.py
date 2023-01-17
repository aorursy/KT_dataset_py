import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import scipy
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import warnings

sns.set_style("whitegrid")
%matplotlib inline
import plotly.offline as pyo
pyo.init_notebook_mode()
from plotly.graph_objs import *
import plotly.graph_objs as go
import json
#__________________
def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#____________________________
def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#_______________________________________
def safe_access(container, index_values):
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan
#_______________________________________
LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews']
#_______________________________________
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',  
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users'}
#_______________________________________     
IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}
#_______________________________________
def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])
#_______________________________________
def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])
#_______________________________________
def convert_to_original_format(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    tmdb_movies['actor_4_name'] = credits['cast'].apply(lambda x: safe_access(x, [4, 'name']))
    tmdb_movies['actor_5_name'] = credits['cast'].apply(lambda x: safe_access(x, [5, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies
credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")
movie = convert_to_original_format(movies, credits)
movie.head()
movie.describe()
corr = movie.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr, vmax=1, square=True)
plt.show()
actor = movie[['actor_1_name', 'actor_2_name', 'actor_3_name', 'actor_4_name', 'actor_5_name', 'gross', 'vote_average', 'num_voted_users', 'popularity']]
actor.head()
actor_list = pd.melt(actor, id_vars=['vote_average', 'num_voted_users'], value_vars=['actor_1_name', 'actor_2_name', 'actor_3_name', 'actor_4_name', 'actor_5_name'],
                    var_name='variable', value_name='actor_name')
actor_list.head()
actor_score = (actor_list['vote_average'] * actor_list['num_voted_users']).groupby(actor_list['actor_name']).sum()
actor_list_gross = pd.melt(actor, id_vars=['gross'], value_vars=['actor_1_name', 'actor_2_name', 'actor_3_name', 'actor_4_name', 'actor_5_name'],
                    var_name='variable', value_name='actor_name')
actor_score_gross= actor_list_gross['gross'].groupby(actor_list_gross['actor_name']).sum()
df = pd.concat([actor_score, actor_score_gross], axis=1)
df.columns = [['vote_average', 'gross']]
warnings.filterwarnings("ignore")

artemis_actors = ['Kenneth Choi', 'Sterling K. Brown', 'Jeff Goldblum', 'Zachary Quinto', 
               'Charlie Day', 'Dave Bautista', 'Sofia Boutella', 'Brian Tyree Henry']

a = df.loc[[i for i in df.index if i not in artemis_actors], :]
b = df.loc[artemis_actors, :]
df = pd.concat([a, b]).dropna()
def quality_graph(df):
    edge_trace = Scatter(
    x=[],
    y=[],
    line = Line(width=0.5,color='#888'),
    hoverinfo = 'none',
    mode = 'lines')

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            line=dict(width=2)))
    
    for ind, col in df.iterrows():
        node_trace['x'] += (col['gross'].values[0], )
        node_trace['y'] += (col['vote_average'].values[0], )
        node_trace['text'] += (ind,)
        if ind in artemis_actors:
            node_trace['marker']['color'] += (10, )
        else:
            node_trace['marker']['color'] += (1, )
        
    fig = Figure(data=Data([node_trace]),
                 layout=Layout(
                    title='<br>Quality of actors',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=XAxis(title='Sum Gross', showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=YAxis(title='IMDB score x Users count', showgrid=True, zeroline=False, showticklabels=True)))    
    return fig