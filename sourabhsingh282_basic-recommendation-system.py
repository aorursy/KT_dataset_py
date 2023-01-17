 # -*- coding: utf-8 -*-

"""

Created on Sun Nov 24 19:04:29 2019



@author: sourabh singh

"""

#this is the recommendatiion system which you see in when you login first time in any account 

#basic  recoomendation system

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcdefaults()

df1 =pd.read_csv('tmdb_5000_credits.csv')

df2 =pd.read_csv('tmdb_5000_movies.csv')

df1.columns=['id','title','cast','crew']

df2=df2.merge(df1,on='id')

y = df2.head(5)

print(y)

C= df2['vote_average'].mean()

C1 = df2['vote_average'].median()



print()

print(C)

m= df2['vote_count'].quantile(0.9)

print()

print(m)

q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape

print(q_movies)

def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    # formula used in imdb rating to rate the movie

    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)



#Print the top 15 movies

z=q_movies[['original_title', 'vote_count', 'vote_average', 'score','popularity']].head(5)

score = z.sort_values('score',ascending = False)

plt.figure(figsize=(5,5))

axis1=sns.barplot(x=score['score'].head(20),y=score['original_title'].head(20))

plt.xlim(4,10)

plt.title('best movie by average votes',weight='bold')

plt.ylabel('score',weight='bold')

plt.ylabel('movie title',weight='bold')

plt.savefig('best_movies.png')



popularity = z.sort_values('popularity',ascending =False)

plt.figure(figsize=(6,5))

ax=sns.barplot(x=popularity['popularity'].head(20),y=popularity['original_title'].head(20),data=popularity)

plt.title('most popular by votes',weight='bold')

plt.xlabel('score of populairty',weight='bold')

plt.ylabel('movie title',weight='bold')

plt.savefig('best_popular_movie.png')

pop= df2.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))



plt.barh(pop['original_title'].head(6),pop['popularity'].head(6), align='center',

        color='skyblue')

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")