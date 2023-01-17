# importing packages 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format
#Loading Data

df_tmbd = pd.read_csv("../input/tmdb-movies.csv")

df_tmbd.head()
df_tmbd.isnull().sum()
df_tmbd.drop(['id','original_title','imdb_id','cast','homepage','director','overview','tagline','keywords','production_companies'], axis = 1, inplace = True)
df_tmbd.isnull().sum()
df_tmbd.dropna(inplace=True, axis = 0)
df_tmbd.isnull().sum()
df_tmbd.duplicated().sum()
df_tmbd.drop_duplicates(inplace = True)



# Number of duplicate rows

df_tmbd.duplicated().sum()
df_tmbd.info()
#Converting Release_date to datetime format

df_tmbd['release_date'] = pd.to_datetime(df_tmbd['release_date'], format = '%m/%d/%y')

df_tmbd.release_date.dtype
df_tmbd['budget_adj'] = df_tmbd['budget_adj'].astype(int)

df_tmbd['budget_adj'] = df_tmbd['revenue_adj'].astype(int)

df_tmbd.info()

df_tmbd.revenue.min()
df_tmbd.head()
df_tmbd.describe()
df_tmbd['budget'].replace({0: df_tmbd['budget'].mean()}, inplace=True)

df_tmbd['revenue'].replace({0: df_tmbd['revenue'].mean()}, inplace=True)
# Number of unique values

df_tmbd.nunique()
df_tmbd.head()
df_tmbd = (df_tmbd.drop('genres', axis=1).join(df_tmbd['genres']

                                               .str.split('|', expand = True).stack().reset_index(level=1, drop=True)

                                               .rename('genres')).loc[:,df_tmbd.columns])
df_tmbd.head()
df_tmbd.shape
genres_revenue = df_tmbd.groupby('genres')['revenue'].mean()

genres_revenue = genres_revenue

genres_revenue
genres_revenue.sort_values(ascending = False).plot(kind='bar', figsize=(12,5));

plt.xlabel('Genres')

plt.ylabel('revenue in 10 million')

plt.title('Average Revenue by genres')
genres_ratings = df_tmbd.groupby('genres')['vote_average'].mean()

genres_ratings = genres_ratings

genres_ratings
genres_ratings.sort_values(ascending=False).plot(kind = 'bar', figsize = (10,5));

plt.xlabel('Genres')

plt.ylabel('Average Ratings')

plt.title('Rating Trends of Genres')
genres_budget = df_tmbd.groupby('genres')['budget'].mean()

genres_budget = genres_budget

genres_budget
genres_budget.sort_values(ascending = False).plot(kind='bar', figsize = (10,5));

plt.xlabel('Genres')

plt.ylabel('budget')

plt.title('Genre By average Budget')
genre_popularity = df_tmbd.groupby('genres')['popularity'].mean()

genre_popularity = genre_popularity

genre_popularity.sort_values(ascending=False).plot(kind='bar', figsize=(8,5));
pd.concat([genres_budget,genres_revenue],axis=1).plot(kind='bar', figsize=(10,4));

plt.xlabel('Genres')

plt.title('Comparison of Budget and revenue')
def scatter_plot(x,y,title):

    df_tmbd.plot(x=x, y=y, kind='scatter',fontsize=8)

    plt.title(title)
scatter_plot("budget","popularity","budget and popularity correlation")
scatter_plot('budget','vote_average', 'Budget and vote_average correlation')
from subprocess import call

call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])