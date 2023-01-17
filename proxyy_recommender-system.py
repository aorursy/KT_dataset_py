# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
df1.head()
df2.head()
df1.columns = ['id','tittle','cast','crew']
df2=df2.merge(df1,on='id')
df2.head()
df2.columns
#DEMOGRAPHIC FILTERING

#mean vote of all movies

C=C= df2['vote_average'].mean()

C
#min votes req to be listed

m= df2['vote_count'].quantile(0.9)

m
q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape
def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

final_movies=q_movies['title'][:10]
final_movies
#content based

def get_director(x):

    for i in x:

        if i['job']=='Director':

            return i['name']

        

    return np.nan    
def get_list(x):

    names=[i['name'] for i in x]

    

    if len(names)>3:

        names=names[:3]

    return names

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(literal_eval)
df2['director']=df2['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(get_list)
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
features = ['cast', 'keywords', 'genres']
def clean_list(x):

    return [str.lower(i.replace(" ","")) for i in x]



def clean_director(x):

    if x=='NaN':

        return ''

    return str.lower(x.replace(" ",""))

    
for feature in features:

    df2[feature]=df2[feature].apply(clean_list)



df2['director']=df2['director'].fillna('NaN')
df2['director']=df2['director'].apply(clean_director)    
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)
df2['soup'].head()
from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df2['soup'])
from sklearn.metrics.pairwise import cosine_similarity



similarity = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()

indices = pd.Series(df2.index, index=df2['title'])
indices.head()
def get_recommendations(title, cosine_sim=similarity):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df2['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises', similarity)
#Collabarative filtering

from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate

reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd=SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)