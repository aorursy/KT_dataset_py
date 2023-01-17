import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credits=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
credits.head()
credits.shape
movies.head()
credits.columns=['id','title','cast','crew']

df_movies=movies.merge(credits, on='id')
df_movies.head()
df_movies.shape
df_movies['overview'].head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer(stop_words = 'english')



df_movies['overview'] = df_movies['overview'].fillna('')



tfv_matrix = tfv.fit_transform(df_movies['overview'])



tfv_matrix
tfv_matrix.shape
from sklearn.metrics.pairwise import linear_kernel



cos_sim = linear_kernel(tfv_matrix, tfv_matrix)



cos_sim
cos_sim.shape
indices = pd.Series(df_movies.index, index = df_movies['original_title']).drop_duplicates()



indices
def get_recommendations(title, cos_sim = cos_sim):

    

    idx=indices[title]

    

    sim_scores = list(enumerate(cos_sim[idx]))

    

    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)

    

    sim_scores = sim_scores[0:15]

    

    movie_indices = [i[0] for i in sim_scores]

    

    return df_movies['original_title'].iloc[movie_indices]
print("Enter the Title Name ")



titlename = 'The Avengers'



print("The list of top 15 recommended movies for the ",titlename ," are as follows")



get_recommendations(titlename)