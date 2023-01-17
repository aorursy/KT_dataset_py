

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline

df_movie = pd.read_csv('../input/tmdb_movies.csv')

df_movie.head()
df_movie.describe()
df_movie.info()
df_movie.query('runtime == 0')['original_title']
df_movie.query('budget_adj == 0')['original_title'].count()
df_movie2 = df_movie.drop(['homepage','imdb_id','tagline','keywords','overview','production_companies','release_date','budget','revenue','cast'], axis=1)
df_movie3 = df_movie2.query('revenue_adj != 0').query('budget_adj != 0')

df_movie3.info()
df_movie3.to_csv('movies_clean.csv', index=False)
df_movie3['genres_2'] = df_movie['genres'].str.split('|').str[0]
df_movie3.drop(['genres'], axis=1, inplace=True)
df_movie3.drop(['id'], axis=1, inplace=True)
df_movie3.genres_2.value_counts()
df_movie3.query("genres_2 == 'TV Movie'")
df_movie3.drop(8615, inplace=True)
df_movie3.vote_average.describe(), df_movie3.popularity.describe()
be_votes = [2.2, 5.7, 6.2, 6.7, 8.4]

be_pop = [0.001117, 0.463068, 0.797723, 1.368403, 32.985763]
bin_names_votes = ['Low', 'Below Average', 'Above Average', 'High']

bin_names_pop = ['Non Popular', 'Semi Popular', 'Popular', 'Very Popular']
df_movie3['rating_level'] = pd.cut(df_movie3['vote_average'], be_votes, labels=bin_names_votes)

df_movie3['pop_level'] = pd.cut(df_movie3['popularity'], be_pop, labels=bin_names_pop)
df_movie3.info()
df_movie3["pop_level"] = df_movie3["pop_level"].astype('object')

df_movie3["rating_level"] = df_movie3["rating_level"].astype('object')
df_movie3.info()
df_movie3.isnull().sum()
df_movie3.dropna(axis=0, inplace=True)
df_movie3.info()
df_movie3.to_csv('movies_clean2.csv', index=False)
sns.relplot(x="popularity", y="revenue_adj", data=df_movie3);
tot_rev = df_movie3.groupby('pop_level')['revenue_adj'].sum()

tot_rev
level_pops = df_movie3.pop_level.unique()

level_pops.sort()
def pie_chart():

    sns.set(context="notebook")

    plt.pie(tot_rev, labels=level_pops, autopct='%1.1f%%')

    plt.legend(title="Popularity",

              loc="center right",

              bbox_to_anchor=(1.5, 0, 0.5, 1));
df_movie3.groupby('genres_2').revenue_adj.describe()

boxplot_data = df_movie3.query('revenue_adj < 500000000')
def boxplot():

    sns.set(style="ticks", font_scale=1.75)

    sns.catplot(x="genres_2", y="revenue_adj", kind="box", height=10, aspect=2, data=boxplot_data)

    plt.xlabel("Genres")

    plt.ylabel("Revenue in $100,000,000s")

    plt.xticks(rotation=45);
pie_chart()
boxplot()