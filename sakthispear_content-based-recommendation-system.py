# Download the data set it is around 160 MB
#!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
#print('unziping ...')
#!unzip -o -j moviedataset.zip
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
# Loading Dateset
movies_df = pd.read_csv('/home/sakthi/Documents/machineLearning/ml-latest/movies.csv')
ratings_df = pd.read_csv('/home/sakthi/Documents/machineLearning/ml-latest/ratings.csv')
movies_df.head()
# Extract Year from title
movies_df['year']= movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
# Remove the Brackets in the Year column
movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)',expand=False)
# Remove the year in title 
movies_df['title']=movies_df.title.str.replace('(\(\d\d\d\d\))','')
# Remove the spaces in title
movies_df['title']=movies_df['title'].apply(lambda x:x.strip())
movies_df.head()
# Split Genres Based on |
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()
# Local copy of Movie Dataframe
moviesWithGenres_df = movies_df.copy()
# Iterate Over each rows on Genres
for index,row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index,genre]=1
# Fill Zeros in place of Not Available values
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()
# Check the Rating Data set
ratings_df.head()
# Remove the Timestamp columns
ratings_df=ratings_df.drop('timestamp',1)
ratings_df.head()
# Getting User Input From user
userInput =[
    {'title':'Breakfast Club, The','rating':5},
    {'title':'Toy Story', 'rating':3.5},
    {'title':'Jumanji','rating':2},
    {'title':'Pulp Fiction','rating':5},
    {'title':'Akira','rating':4.5}
]
# Convert the data into dataframe
inputMovies = pd.DataFrame(userInput)
inputMovies
# Getting the list (Movie Id, Title, Genre, Year) of user inputted Movies
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Merge 
inputMovies =pd.merge(inputId, inputMovies)
# Remove the genre and year columns
inputMovies = inputMovies.drop('genres',1).drop('year',1)
inputMovies
# Find the UserMovie id with genre rating
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies
# Reset the Index 
userMovies = userMovies.reset_index(drop = True)
# Remove the unwanted Columns
userGenreTable = userMovies.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
userGenreTable
# User Rating
inputMovies['rating']
# Element wise multiplication of 
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userProfile
# Set Index of MovieId
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
genreTable.head()
genreTable.shape
# Recommendation for 
recommendationTable_df = ((genreTable * userProfile) .sum(axis = 1)) / (userProfile.sum())
recommendationTable_df.head()
# Sort the value
recommendationTable_df = recommendationTable_df.sort_values(ascending = False)
recommendationTable_df.head()
# Find the Corresponding Movie for each movie id
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
