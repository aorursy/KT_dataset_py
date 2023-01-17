#__________________

import json

import pandas as pd

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

    # I'm assuming that the first production country is equivalent, but have not been able to validate this

    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['director_name'] = credits['crew'].apply(get_director)

    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))

    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))

    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))

    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)

    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)

    return tmdb_movies
#______________

# the packages

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

#___________________

# and the dataframe

credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")

movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")

df = convert_to_original_format(movies, credits)

#___________________________

# countries in the dataframe

df['country'].unique()
df_countries = df['title_year'].groupby(df['country']).count()

df_countries = df_countries.reset_index()

df_countries.rename(columns ={'title_year':'count'}, inplace = True)

df_countries = df_countries.sort_values('count', ascending = False)

df_countries.reset_index(drop=True, inplace = True)
sns.set_context("poster", font_scale=0.6)

plt.rc('font', weight='bold')

f, ax = plt.subplots(figsize=(11, 6))

labels = [s[0] if s[1] > 80 else ' ' 

          for index, s in  df_countries[['country', 'count']].iterrows()]

sizes  = df_countries['count'].values

explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(df_countries))]

ax.pie(sizes, explode = explode, labels = labels,

       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',

       shadow=False, startangle=45)

ax.axis('equal')

ax.set_title('% of films per country',

             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);

data = dict(type='choropleth',

locations = df_countries['country'],

locationmode = 'country names', z = df_countries['count'],

text = df_countries['country'], colorbar = {'title':'Films nb.'},

colorscale=[[0, 'rgb(224,255,255)'],

            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],

            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],

            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],

            [1, 'rgb(227,26,28)']],    

reversescale = False)
layout = dict(title='Number of films in the TMDB database',

geo = dict(showframe = True, projection={'type':'Mercator'}))
choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)