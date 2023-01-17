import numpy as np

import pandas as pd
ratings = pd.read_csv('../input/ratings.csv')

ratings.head()
movies = pd.read_csv('../input/movies.csv')

movies.head()
data = ratings.merge(movies, on='movieId', how ='left')

data.head()
#Feature Engineering

average_ratings = pd.DataFrame(data.groupby('title')['rating'].mean())

average_ratings.head()
#Total Number Of Rating 

average_ratings['Total Ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())

average_ratings.head()
#build the Recommender 

movie_user = data.pivot_table(index = 'userId', columns = 'title', values = 'rating')

movie_user.head()
correlations = movie_user.corrwith(movie_user['Toy Story (1995)'])

correlations.head()
recommendation = pd.DataFrame(correlations,columns=['Correlation'])

recommendation.dropna(inplace=True)

recommendation = recommendation.join(average_ratings['Total Ratings'])

recommendation.head()
recc = recommendation[recommendation['Total Ratings']>100].sort_values('Correlation',ascending=False).reset_index()
recc = recc.merge(movies,on='title', how='left')

recc.head(10)