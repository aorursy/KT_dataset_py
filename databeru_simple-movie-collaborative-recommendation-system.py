# Let's have a look at the csv-files
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/movielens-20m-dataset"))
rating = pd.read_csv('../input/movielens-20m-dataset/rating.csv')
rating.shape
# rating has around 20.000.000 ratings. This is huge.
# Therefore we will use only 50% of the ratings to gain some calculating speed
rating = rating[:10000000]
rating.shape
rating.head()
rating = rating[rating.columns.drop("timestamp")]
rating.describe()
rating.info()
plt.figure(figsize=(8,6))
sns.heatmap(rating.isnull())
plt.title("Missing values?", fontsize = 15)
plt.show()
movie = pd.read_csv('../input/movielens-20m-dataset/movie.csv')
movie.head(5)
movie.describe()
movie.info()
movie.head()
# Merge both DataFrame to have also the titles of the movies
df = pd.merge(movie,rating)

# Keep only the columns title, userId and rating
df = df[['title','userId','rating']]
# Show the result
df.head()
# Group the titles by number of ratings to see which movies where rated the most
count_rating = df.groupby("title")['rating'].count().sort_values(ascending=False)
count_rating.head(10)
# Select the movies with at least 500 ratings
r = 500
more_than_200_ratings = count_rating[count_rating.apply(lambda x: x >= r)].index

# Keep only the movies with at least 500 ratings in the DataFrame
df_r = df[df['title'].apply(lambda x: x in more_than_200_ratings)]
# Display the count of ratings the each movie
# Having only the movies with at least 500 ratings
df_r.groupby("title")['rating'].count().sort_values(ascending=False)
before = len(df.title.unique())
after = len(df_r.title.unique())
rows_before = df.shape[0]
rows_after = df_r.shape[0]
print(f'''There are {before} movies in the dataset before filtering and {after} movies after the filtering.

{before} movies => {after} movies
{rows_before} rows before filtering => {rows_after} rows after filtering''')
# Create a matrix with userId as rows and the titles of the movies as column.
# Each cell will have the rating given by the user to the movie.
# There will be a lot of NaN values, because each user hasn't watched most of the movies
movies = df_r.pivot_table(index='userId',columns='title',values='rating')
movies.iloc[:5,:5]
# Let's choose a famous movie
movie = 'Jurassic Park (1993)'

# Display the first ratings of the users for this movies
movies[movie].head(5)
def find_corr(df_movies, movie_name):
    '''
    Get the correlation of one movie with the others
    
    Args
        df_movies (DataFrame):  with user_id as rows and movie titles as column and ratings as values
        movie_name (str): Name of the movie
    
    Return
        DataFrame with the correlation of the movie with all others
    '''
    
    similar_to_movie = df_movies.corrwith(movies[movie_name])
    similar_to_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    similar_to_movie = similar_to_movie.sort_values(by = 'Correlation', ascending = False)
    return similar_to_movie
# Let's try with the first movie
movie_name = 'Jurassic Park (1993)'
find_corr(movies, movie_name)
# Let's try with the first movie
movie_name = 'Terminator 2: Judgment Day (1991)'
find_corr(movies, movie_name)