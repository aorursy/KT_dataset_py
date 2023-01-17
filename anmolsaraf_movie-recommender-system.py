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
import pandas as pd

tmdb_5000_credits = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

tmdb_5000_movies = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
tmdb_5000_credits.head()
tmdb_5000_credits.shape
tmdb_5000_movies.head()
tmdb_5000_movies.shape
tmdb_5000_credits.rename(columns={"movie_id":"id"}, inplace=True)

tmdb_5000_credits.head()
merged_tmdb=pd.merge(tmdb_5000_credits, tmdb_5000_movies, on='id')

print(merged_tmdb.shape)

merged_tmdb.head()
tmdb_final=merged_tmdb.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])

tmdb_final.head()
V = tmdb_final['vote_count']

R = tmdb_final['vote_average']

C = tmdb_final['vote_average'].mean()

m = tmdb_final['vote_count'].quantile(0.70)



tmdb_final['weighted_average'] = (V/(V+m) * R) + (m/(m+V) * C)

print("C= %s, m= %s"%(C,m))



#Displaying the first 5 rows.

tmdb_final.loc[:4,['original_title','weighted_average','vote_average']]
tmdb_movies_ranked = tmdb_final.sort_values('weighted_average', ascending=False)

tmdb_movies_ranked[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(11)
import matplotlib.pyplot as plt

import seaborn as sns



wavg = tmdb_movies_ranked



plt.figure(figsize=(16,6))



ax = sns.barplot(x=wavg['weighted_average'].head(20), y=wavg['original_title'].head(20), data=wavg, palette='deep')



plt.xlim(7, 8.35)

plt.title('"Best" Movies by TMDB Votes', weight='bold')

plt.xlabel('Weighted Average Score', weight='bold')

plt.ylabel('Movie Title', weight='bold')

plt.show()
popular = tmdb_movies_ranked.sort_values('popularity', ascending=False)



plt.figure(figsize=(16,6))



ax = sns.barplot(x=popular['popularity'].head(20), y=popular['original_title'].head(20), data=popular, palette='deep')



plt.title('"Most Popular" Movies by TMDB Votes', weight='bold')

plt.xlabel('Popularity Score', weight='bold')

plt.ylabel('Movie Title', weight='bold')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer



# Using Abhishek Thakur's arguments for TF-IDF

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Filling NaNs with empty string

tmdb_final['overview'] = tmdb_final['overview'].fillna('')



# Fitting the TF-IDF on the 'overview' text

tfv_matrix = tfv.fit_transform(tmdb_final['overview'])



tfv_matrix.shape
from sklearn.metrics.pairwise import linear_kernel



# Compute the linear kernel

lin_k = linear_kernel(tfv_matrix, tfv_matrix)



# Reverse mapping of indices and movie titles

indices = pd.Series(tmdb_final.index, index=tmdb_final['original_title']).drop_duplicates()

# Credit to Ibtesam Ahmed for the skeleton code

def give_rec(title, lin_k=lin_k):

    # Get the index corresponding to original_title

    idx = indices[title]



    # Get the pairwsie similarity scores 

    lin_k_scores = list(enumerate(lin_k[idx]))



    # Sort the movies 

    lin_k_scores = sorted(lin_k_scores, key=lambda x: x[1], reverse=True)



    # Scores of the 10 most similar movies

    lin_k_scores = lin_k_scores[1:11]



    # Movie indices

    movie_indices = [i[0] for i in lin_k_scores]



    # Top 10 most similar movies

    return tmdb_final['original_title'].iloc[movie_indices]
give_rec('The Dark Knight')