import pandas as pd

import numpy as np



from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer



import json

from functools import reduce
credits = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
# Dataset shape

print("Credits shape is {}".format(credits.shape))

print("Movies shape is {}".format(movies.shape))

print(credits.columns)

print(movies.columns)
credits.head()
movies.head()
final_dataset = pd.merge(movies,credits,left_on='id',right_on='movie_id',how='left')
final_dataset.isnull().sum() 
# drop homepage and release_date

final_dataset.drop(['homepage','release_date','runtime'],axis=1,inplace=True)
final_dataset['overview'].fillna('',inplace=True)

final_dataset['tagline'].fillna('',inplace=True)
final_dataset.isnull().sum() 
# some of the columns are given in JSON format, We should convert this to Dictinary using json.loads method



def convertJson(y):

    y = json.loads(y)

    return " ".join([val['name'] for val in y])

final_dataset['genres'] = final_dataset['genres'].apply(convertJson)

final_dataset['keywords'] = final_dataset['keywords'].apply(convertJson)

final_dataset['production_companies'] = final_dataset['production_companies'].apply(convertJson)

final_dataset['production_countries'] = final_dataset['production_countries'].apply(convertJson)
final_dataset.drop(['id','spoken_languages','status','budget','popularity','revenue','vote_average','vote_count','crew'],inplace=True,axis=1)
final_dataset['genres']
# Top 5 cast does better prediction

def get_cast(y):

    y = json.loads(y)

    return " ".join([val['character']+" "+ val['name'] for val in y[:5]])

final_dataset['cast'] = final_dataset['cast'].apply(get_cast)
columns = ['original_language','original_title','overview',\

              'production_countries','tagline','title_x','title_y','cast']

final_dataset['title'] = final_dataset['title_x']

final_dataset['keywords'] = final_dataset[['keywords','genres','production_companies'] + columns].apply(" ".join,axis=1)

final_dataset.drop(columns,inplace=True,axis=1)
final_dataset.head()
# stop words will remove the common english words like a,an,the,i,me,my etc which increase the words count and 

# create noise in our model 



c_vect = TfidfVectorizer()

X = c_vect.fit_transform(final_dataset['keywords'])
# There are other similiary distance metric available which are euclidean distance,manhattan distance, 

# Pearson coefficient etc. But for sparse matrix cosine similarity works better

cosine_sim = cosine_similarity(X)
def get_movie_recommendation(movie_name):

    idx = final_dataset[final_dataset['title'].str.contains(movie_name)].index

    if len(idx):

        sorted_list_indices = sorted(list(enumerate(cosine_sim[idx[0]])), key=lambda x: x[1], reverse=True)[1:11]

        sorted_list_indices = list(map(lambda x:x[0],sorted_list_indices))

        return sorted_list_indices

    else : 

        return []
title = "The Avengers"

recommended_movie_list = get_movie_recommendation(title)

final_dataset.loc[recommended_movie_list,['title','genres']]
final_dataset.loc[[3, 65, 3854]]
title = "The Dark Knight Rises"

recommended_movie_list = get_movie_recommendation(title)

final_dataset.loc[recommended_movie_list,['title','genres']]