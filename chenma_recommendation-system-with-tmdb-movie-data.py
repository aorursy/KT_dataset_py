import numpy as np

import pandas as pd

from functools import reduce

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import math, nltk, warnings

from nltk.corpus import wordnet

from sklearn import linear_model

from sklearn.neighbors import NearestNeighbors
import json



#___________________________

def load_tmdb_movies(path):

    df = pd.read_csv(path)

    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())

    json_columns = ['genres', 'keywords', 'production_countries',

                    'production_companies', 'spoken_languages']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df

#___________________________

def load_tmdb_credits(path):

    df = pd.read_csv(path)

    json_columns = ['cast', 'crew']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df

#___________________

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

#____________________________________

TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {

    'budget': 'budget',

    'genres': 'genres',

    'revenue': 'gross',

    'title': 'movie_title',

    'runtime': 'duration',

    'original_language': 'language',

    'keywords': 'plot_keywords',

    'vote_count': 'num_voted_users'}

#_____________________________________________________

IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}

#_____________________________________________________

def safe_access(container, index_values):

    # return missing value rather than an error upon indexing/key failure

    result = container

    try:

        for idx in index_values:

            result = result[idx]

        return result

    except IndexError or KeyError:

        return pd.np.nan

#_____________________________________________________

def get_director(crew_data):

    directors = [x['name'] for x in crew_data if x['job'] == 'Director']

    return safe_access(directors, [0])

#_____________________________________________________

def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])

#_____________________________________________________

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
plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "last_expr"

pd.options.display.max_columns = 50

%matplotlib inline

warnings.filterwarnings('ignore')

PS = nltk.stem.PorterStemmer()
from subprocess import check_output

print(check_output(["ls","../input/"]))
# load the dataset

#credits = load_tmdb_credits("/Users/machen/Documents/000 MSBA/452 Machine Learning/HW3 TMDB MOVIE/tmdb/tmdb_5000_credits.csv")

#movies = load_tmdb_movies("/Users/machen/Documents/000 MSBA/452 Machine Learning/HW3 TMDB MOVIE/tmdb/tmdb_5000_movies.csv")

credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")

movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")

df_initial = convert_to_original_format(movies, credits)

df_initial = convert_to_original_format(movies, credits)

print('Shape:',df_initial.shape)
# info on variable types and filling factor

tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values'}))

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

tab_info
C = df_initial['vote_average'].mean()

C
m = df_initial['num_voted_users'].quantile(q=0.9)

m
movies = df_initial.copy().loc[df_initial['num_voted_users']>= m]

movies.shape
def weighted_rating(x, m=m, C=C):

    v = x['num_voted_users']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`

movies['weighted_score'] = movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above

movies = movies.sort_values(by = 'weighted_score', ascending=False, axis = 0)



#Print the top 10 movies

movies[['movie_title','weighted_score','title_year','director_name']].head(10)
df_initial['genres'] = df_initial['genres'].str.split('|')

df_initial['plot_keywords'] = df_initial['plot_keywords'].str.split('|')
features = ['genres', 'plot_keywords', 'director_name', 'actor_1_name','actor_2_name','actor_3_name']

df_initial[features].head()
# Function to convert all strings to lower case and strip names of spaces

def clean_data(x):

    #for genres and plot_keywords

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        #for director, actors

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''
# Apply clean_data function to your features.

for feature in features:

    df_initial[feature] = df_initial[feature].apply(clean_data)
df_initial[features].head()
# Function to join all attributes

def create_soup(x):

    return ' '.join(x['plot_keywords']) + ' ' + ' '.join(x['actor_1_name']) + ' ' + ' '.join(x['director_name']) + ' ' + ' '.join(x['genres'])+ ' ' + ' '.join(x['actor_2_name'])+ ' ' + ' '.join(x['actor_3_name'])
# Apply create_soup function to your df

df_initial['soup'] = df_initial.apply(create_soup, axis =1)

df_initial['soup'].head()
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df_initial['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim1 = cosine_similarity(count_matrix, count_matrix)
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

df_initial['overview'] = df_initial['overview'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(df_initial['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape



# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Reset index of our main DataFrame and construct reverse mapping as before

df_initial = df_initial.reset_index()

indices = pd.Series(df_initial.index, index=df_initial['movie_title'])

indices.head()
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return df_initial['movie_title'].iloc[movie_indices]
get_recommendations('The Shawshank Redemption')
get_recommendations('The Shawshank Redemption',cosine_sim1)