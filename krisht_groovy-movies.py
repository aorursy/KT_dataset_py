import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
#Import all the data from the csv

movies = pd.read_csv('../input/movie_metadata.csv')

print("Sample Size:", movies.shape[0])

print("Samples' Attributes:", movies.shape[1])
movies.describe()
impColums = ['num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross','num_voted_users','cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews','budget','title_year','actor_2_facebook_likes','imdb_score','aspect_ratio','movie_facebook_likes']
corrmat = movies[impColums].corr()

sns.heatmap(corrmat, vmax=.8, square=True)
cluster_selected = ['movie_facebook_likes', 'imdb_score', 'title_year', 'gross']

clusterNoNA = movies[cluster_selected].dropna()



print("Number of observations:", clusterNoNA.shape[0]);
sns.pairplot(clusterNoNA)
sns.distplot(movies['title_year'].dropna())
sns.distplot(movies['duration'].dropna())
plt.plot(movies['budget'], movies['gross'],'.')
movies['profit']= movies.gross-movies.budget

top10fails  = movies.sort(columns='profit').head(10); 

top10hits = movies.sort(columns='profit', ascending = False).head(10); 

top10fails[['gross', 'budget']].groupby(movies['movie_title']).sum().plot.barh(stacked=True, title='Top-15 fails')
top10hits[['budget', 'gross']].groupby(movies['movie_title']).sum().plot.barh(stacked=True, 

                                                                              title='Top-15 success')
meanScore = movies.imdb_score.groupby(movies.director_name).mean().sort_values(ascending=False)

meanScore[:10].sort_values().plot.barh(figsize=(6,8), title='Top 10 Directors with the Highest Score');
meanScore = movies.imdb_score.groupby(movies.director_name).mean().sort_values()

meanScore[:10].sort_values().plot.barh(figsize=(6,8), title = 'Bottom 10 Directors with Least Scores')