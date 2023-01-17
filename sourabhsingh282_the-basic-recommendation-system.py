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
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

plt.rcdefaults()

df1 = pd.read_csv('../input/tmdb-5000-creditscsv/tmdb_5000_credits.csv')

df2 =pd.read_csv('../input/movie-recommendation-system-data-set/tmdb_5000_movies.csv')
df1.columns=['id','title','cast','crew']

df2=df2.merge(df1,on='id')

y = df2.head(5)

print(y)
C = df2['vote_average'].mean()

print(C)
C1 = df2['vote_average'].median()

print(C1)
m = df2['vote_count'].quantile(0.9)

print(m)
q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape

print(q_movies)
def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

print(q_movies)
q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies)
z=q_movies[['original_title', 'vote_count', 'vote_average', 'score','popularity']].head(5)

print(z)
score = z.sort_values('score',ascending = False)

print(score)
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
plt.figure(figsize=(12,4))



plt.barh(pop['original_title'].head(6),pop['popularity'].head(6), align='center',

        color='skyblue')

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")