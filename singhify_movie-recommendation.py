import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings





warnings. filterwarnings('ignore')

pd.set_option('display.max_columns',None)

credit = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv') 
credit.head()
movies.head()
print('Credit:',credit.shape)

print('Movie:',movies.shape)
credit = credit.rename(index=str,columns={'movie_id':'id'})

credit.columns
movies.columns
###### merging the two datasets



movie_merge = pd.merge(credit,movies,on='id')

movie_merge.head()
movie_merge.shape
movie_merge.columns
movie_merge['status'].unique()
movie_merge_clean = movie_merge.drop(['title_x','homepage','production_countries','title_y','status'],axis=1)
movie_merge_clean.info()
C = movie_merge_clean['vote_average'].mean()
v= movie_merge_clean['vote_count']

R= movie_merge_clean['vote_average']

C= C

m = movie_merge_clean['vote_count'].quantile(.7)
movie_merge_clean['weighted_average'] = ((R*v)+(C*m))/(v+m)
movie_merge_clean.head()
movie_rank = movie_merge_clean.sort_values('weighted_average',ascending = False)

movie_rank.head(10)
movie_merge_clean.columns
movie_rank[['id','original_language','original_title','popularity','vote_average','vote_count','weighted_average']].head(10)
wa = movie_rank.sort_values('weighted_average',ascending=False)

plt.figure(figsize=(10,8))

sns.barplot(y=wa['original_title'].head(15),x=wa['weighted_average'].head(15))

plt.title('Best movie by average weights')
pop = movie_rank.sort_values('popularity',ascending=False)

plt.figure(figsize=(10,8))

sns.barplot(y=pop['original_title'].head(15),x=pop['popularity'].head(15))

plt.title('Best movie by Popularity')
from sklearn.preprocessing import MinMaxScaler



scaled_movies= MinMaxScaler().fit_transform(movie_merge_clean[['weighted_average','popularity']])

normalised_movie = pd.DataFrame(scaled_movies,columns=['Weighted Average','Popularity'])

normalised_movie
movie_rank[['Normalised Weights','Normalised Population']] = normalised_movie

movie_rank
movie_rank['Score'] = movie_rank['Normalised Weights']*.5 + movie_rank['Normalised Population']*.5

movie_rank = movie_rank.sort_values(['Score'],ascending=False)

movie_rank
movie_rank[['original_title','weighted_average','vote_count','vote_average','Score','Normalised Weights','Normalised Population']].sort_values('Score',ascending=False)
Score = movie_rank.sort_values('Score',ascending=False)

plt.figure(figsize=(10,8))

sns.barplot(y=Score['original_title'].head(10),x=Score['Score'].head(10))

plt.title('Best movie by Score')