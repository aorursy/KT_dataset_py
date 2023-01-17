import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)

pd.options.mode.chained_assignment = None



movies_df = pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')

ratings_df = pd.read_csv('../input/imdb-extensive-dataset/IMDb ratings.csv')

movies_df = movies_df.merge(ratings_df, on='imdb_title_id', how='inner')



movies_df.head()
movies_df = movies_df[movies_df.date_published.notnull()]

movies_df['date_published'] = pd.to_datetime(movies_df['date_published'])



sns.set(style="white")



plt.figure(figsize=(15,10))

plt.title('Movies by the year', size=20)

sns.distplot(movies_df.year, bins=100, kde=False)

plt.ylabel('Number of movies', size=15)

plt.xlabel('Year of release',size=15)

plt.axis([1920, 2019, 0, 5000])

plt.xticks(np.arange(1920, 2019, step=5),rotation=45, ha='right')

plt.show()
movies_df = movies_df[movies_df.avg_vote.notnull()]

sns.jointplot(x=movies_df['year'], y=movies_df['avg_vote'],

              kind="kde").fig.set_size_inches(15,15)
a = plt.cm.cool



plt.figure(figsize=(15,10))

count = movies_df['production_company'].value_counts()[:10]

sns.barplot(count.values, count.index, palette=[a(0.1),a(0.2),a(0.3),a(0.4),a(0.5),a(0.6),a(0.7),a(0.8),a(0.9),a(0.99)])

for i, v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=14)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Production Company name', fontsize=12)

plt.title("Distribution of Studio names", fontsize=16)
movies_df['first_genre'] = movies_df['genre'].str.split(',').str[0]



a = plt.cm.cool



plt.figure(figsize=(15,10))

count = movies_df['first_genre'].value_counts()[:7]

sns.barplot(count.values, count.index, palette=[a(0.1),a(0.2),a(0.3),a(0.4),a(0.5),a(0.6),a(0.7)])

for i, v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=14)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Genre name', fontsize=12)

plt.title("Distribution of Genres", fontsize=16)
top_genres = list(count.index)

movie_genres_df = movies_df[movies_df['first_genre'].isin(top_genres)]

movie_genres_df = movie_genres_df[pd.notnull(movie_genres_df[['first_genre', 'avg_vote', 'votes']])]



plt.figure(figsize=(15, 10))

sns.boxplot(x='first_genre', y='avg_vote', data=movie_genres_df)

plt.xlabel("Genre Name",fontsize=12)

plt.ylabel("IMDb Average Rating",fontsize=12)

plt.title("Boxplot of average rating per Genre", fontsize=16)

plt.show()