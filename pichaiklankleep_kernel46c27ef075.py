import pandas as pd 

import numpy as np 

df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

df1.columns = ['id','tittle','cast','crew']

df2= df2.merge(df1,on='id')
df2.head(5)
C= df2['vote_average'].mean()

C
m= df2['vote_count'].quantile(0.9)

m
q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape
def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
pop= df2.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))



plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',

        color='skyblue')

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")