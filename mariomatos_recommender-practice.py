# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



df1 = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df2 = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')



df1 = df1.rename({'movie_id': 'id'}, axis=1)

df1 = df1.drop(['title'], axis=1)



df2 = df2.merge(df1, on='id')



df2.head()



df2.columns
C = df2['vote_average'].mean()

m = df2['vote_count'].quantile(0.9)

C, m
# Filter out movies that don't have 90 % of vote count

q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape
def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)



q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)



q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
pop = df2.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))



plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center') 

plt.gca().invert_yaxis()

plt.xlabel('Popularity')

plt.title('Popular Movies')
df2['overview'].head(5)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import linear_kernel

from sklearn.metrics.pairwise import cosine_similarity



tfidf = TfidfVectorizer(stop_words='english')



df2['overview'] = df2['overview'].fillna('')



tfidf_matrix = tfidf.fit_transform(df2['overview'])



tfidf_matrix.shape



cosine_sim = linear_kernel(tfidf_matrix)
# Trying cosine similarity



documents = [

    'alpine snow winter boots.',

    'snow winter jacket.',

    'active swimming briefs',

    'active running shorts',

    'alpine winter gloves'

]



cntvt = CountVectorizer(stop_words='english')



tfidf_matrix = cntvt.fit_transform(documents)

cntvt.get_feature_names()

tfidf_matrix.todense()



cos_sim = cosine_similarity(tfidf_matrix)

cos_sim

indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()



def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df2['title'].iloc[movie_indices]



idx = indices["The Dark Knight Rises"]

df2['title'].iloc[[i[0] for i in (sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11])]]
get_recommendations('The Dark Knight Rises')
get_recommendations('The Avengers')
# literal_eval is a python function to evaluate correctness of string data. It

# will also create python objects for you 

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']



df2['cast'][0]

df2['crew'][0]

df2['keywords'][0]

df2['genres'][0]



for feature in features:

    df2[feature] = df2[feature].apply(literal_eval)
def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
# return top 3

def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        if len(names) > 3:

            names = names[:3]

        return names

    

    return []
df2['director'] = df2['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(get_list)

    

df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
#data cleaning and prepa



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

    df2[feature] = df2[feature].apply(clean_data)
# create "soup" for the vectorization used to compute the cosine similarity matrix



def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df2['soup'])



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()

indices = pd.Series(df2.index, index=df2['title'])
get_recommendations('The Dark Knight Rises', cosine_sim2)
get_recommendations('The Godfather', cosine_sim2)
# User-User, Item-Item Collaborative filtering



from surprise import Reader, Dataset, KNNBasic

from surprise.model_selection import cross_validate, KFold



reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()



data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)



svd = KNNBasic()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



trainset = data.build_full_trainset()

svd.fit(trainset)



ratings[ratings['userId'] == 1]
svd.predict(1, 3671)