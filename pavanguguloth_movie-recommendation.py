# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
movies=pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credits=pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

credits.columns=['id','title','cast','crew']

movies=movies.merge(credits,on='id')

movies=movies.drop(columns=['title_y','runtime','homepage'],axis=1)

movies.rename(columns={'title_x':'title'},inplace=True)

movies.head()
movie_ratings=movies[['title','vote_average','vote_count']]

movie_ratings.head()
plt.figure(figsize =(10, 4)) 

  

movie_ratings['vote_average'].hist(bins = 70) 
mean_average=movie_ratings['vote_average'].mean()

mean_average

threshold= movie_ratings['vote_count'].quantile(0.9)

def popularity(x,m=threshold,c=mean_average):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * c)
movie_ratings['weighted_average']=movie_ratings.apply(popularity,axis=1)

movie_ratings=movie_ratings.sort_values('weighted_average',ascending=False)

movie_ratings.head()
movie_ratings = movie_ratings.reset_index(drop=True)

indices = pd.Series(movie_ratings.index, index=movie_ratings['title'])



#movie_ratings=movie_ratings.drop('level_0',axis=1)

movie_ratings.head()
def get_top_five_movies():

    return movie_ratings.title.head(5)
get_top_five_movies()
from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    movies[feature] = movies[feature].apply(literal_eval)

    
movies.head()
def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan


movies['director'] = movies['crew'].apply(get_director)

movies['director'].head(3)
def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        if len(names) > 3:

            names = names[:3]

        return names

    return []
features = ['cast', 'keywords', 'genres']

for feature in features:

    movies[feature] = movies[feature].apply(get_list)

movies[['title', 'cast', 'director', 'keywords', 'genres']].head(3)    

    
def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''
features = ['cast', 'keywords', 'director', 'genres']



for feature in features:

    movies[feature] = movies[feature].apply(clean_data)

movies[features].head(3)    
def combine_features(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies['combined_feature'] = movies.apply(combine_features, axis=1)

movies['combined_feature'].head(3)
count_matrix=CountVectorizer().fit_transform(movies['combined_feature'])

count_matrix.shape
cos_similarity=cosine_similarity(count_matrix,count_matrix)

cos_similarity.shape
def get_index_from_title(title):

    return movies[movies.title == title]["index"].values[0]



def get_title_from_index(index):

    return movies[movies.index == index]["title"].values[0]

movies = movies.reset_index()

indices = pd.Series(movies.index, index=movies['title'])
def get_top_five_recommendations(movie_user_likes):

    movie_index = get_index_from_title(movie_user_likes)

    similar_movies =  list(enumerate(cos_similarity[movie_index]))

    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

    i=0

    print("Top 5 similar movies to "+movie_user_likes+" are:")

    for element in sorted_similar_movies:

        print(get_title_from_index(element[0]))

        i=i+1

        if i>=5:

            break

    
get_top_five_recommendations('The Hobbit: The Desolation of Smaug')