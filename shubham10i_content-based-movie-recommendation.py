import numpy as np
import pandas as pd
credits = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits.head()
movies.head()
print("Credits Shape: ",credits.shape)
print("Movies Shape: ",movies.shape)
credit_renamed = credits.rename(index=str,columns={'movie_id':'id'})
movies_df = movies.merge(credit_renamed,on='id')
movies_df.head()
movies_df.columns
movies_df = movies_df.drop(columns=['homepage','title_x','title_y','production_countries'])
movies_df.head()
movies_df.columns
movies_df.info()
movies_df.head(2)['overview']
from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=3, max_features=None,
                     strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
                     ngram_range=(1,3),
                     stop_words='english')

movies_df['overview'] = movies_df['overview'].fillna('')
tfv_matrix = tfv.fit_transform(movies_df['overview'])
print(tfv_matrix[0])
tfv_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)
sig[0]
indices = pd.Series(movies_df.index,index=movies_df['original_title']).drop_duplicates()
indices
indices['Newlyweds']
sig[4799]
list(enumerate(sig[indices['Newlyweds']]))
sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True)
def give_rec(title,sig=sig):
    idx = indices[title]
    sig_score = list(enumerate(sig[idx]))
    sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
    sig_score = sig_score[1:11]
    movie_indices = [i[0] for i in sig_score]
    return movies_df['original_title'].iloc[movie_indices]
give_rec('Avatar')
give_rec('The Dark Knight')
