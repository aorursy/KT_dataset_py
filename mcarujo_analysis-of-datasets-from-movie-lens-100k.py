import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Importing all the rating and the movies names to be merge in the future

ratings = pd.read_csv("../input/movielens-100k-small-dataset/ratings.csv")

ratings = ratings.drop(['timestamp'], axis=1)

movies = pd.read_csv("../input/movielens-100k-small-dataset/movies.csv")

movies = movies.drop(['genres'],axis=1)
ratings.head(10)
# Exemple and Statistic

print(f"Rating options which were used: {sorted(ratings['rating'].unique())}")

print(f"Mean {ratings['rating'].mean()}")

print(f"Standard Deviation {ratings['rating'].std()}")



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5)) 

plt.figure(figsize=(10, 5))

sns.boxplot(x=ratings['rating'],linewidth=2.5,width=0.5, ax=ax1)

sns.distplot(ratings['rating'],axlabel='Histogram of all ratings', ax=ax2)

f.show()

id_top_movies = ratings['movieId'].value_counts()[:10].index

df_names_top_movies = movies[movies.movieId.isin(id_top_movies)]

df_ratings_top_movies = ratings[ratings.movieId.isin(id_top_movies)]

df_ratings_top_movies = df_ratings_top_movies.merge(df_names_top_movies,left_on='movieId', right_on='movieId')

plt.figure(figsize=(30, 5))

sns.boxplot(x="title",

            y="rating",

            palette=["m", "g"],

            data=df_ratings_top_movies,width=0.5).set(xlabel='Movies',ylabel='Rate')
df_ratings_top_movies

id_top_user = ratings['userId'].value_counts()[:10].index

id_top_user
id_top_user = ratings['userId'].value_counts()[:10].index

df_ratings_top_users = ratings[ratings.userId.isin(id_top_user)]

plt.figure(figsize=(30, 5))

sns.boxplot(x="userId",

            y="rating",

            palette=["m", "g"],

            data=df_ratings_top_users,width=0.5).set(xlabel='Users',ylabel='Rate')