!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip

print('unziping ...')

!unzip -o -j moviedataset.zip 
#Dataframe manipulation library

import pandas as pd

#Math functions, we'll only need the sqrt function so let's import only that

from math import sqrt

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#Storing the movie information into a pandas dataframe

movies_df = pd.read_csv('movies.csv')

#Storing the user information into a pandas dataframe

ratings_df = pd.read_csv('ratings.csv')

#Head is a function that gets the first N rows of a dataframe. N's default is 5.

movies_df.head()
ratings_df.shape
#Using regular expressions to find a year stored between parentheses

#We specify the parantheses so we don't conflict with movies that have years in their titles

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

#Removing the parentheses

movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)

#Removing the years from the 'title' column

movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

#Applying the strip function to get rid of any ending whitespace characters that may have appeared

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#movies_df['title']=movies_df['title'].apply(lambda x: x.strip())

movies_df.shape
#Every genre is separated by a | so we simply have to call the split function on |

movies_df['genres'] = movies_df.genres.str.split('|')

movies_df.head()
#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.

moviesWithGenres_df = movies_df.copy()



#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column

for index, row in movies_df.iterrows():

    for genre in row['genres']:

        moviesWithGenres_df.at[index, genre] = 1

#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre

moviesWithGenres_df = moviesWithGenres_df.fillna(0)

moviesWithGenres_df.head()
moviesWithGenres_df.shape
#Drop removes a specified row or column from a dataframe

ratings_df = ratings_df.drop('timestamp', 1)

ratings_df.head()
userInput_Ferdous = [

            {'title':'Shawshank Redemption, The', 'rating':5},

            {'title':'Dark Knight, The', 'rating':4.8},

            {'title':'Forrest Gump', 'rating':5},

            {'title':"Godfather, The", 'rating':5},

            {'title':'Hangover, The', 'rating':3.5},           

            {'title':'Pulp Fiction', 'rating':4},

            {'title':"One Flew Over the Cuckoo's Nest", 'rating':5},

            {'title':'Interstellar', 'rating':4},

            {'title':'Death Race', 'rating':4.5},

            {'title':'City of God', 'rating':5},

            {'title':"Schindler's List", 'rating':5},

            {'title':'Doctor Strange', 'rating':4},

            {'title':'Saving Private Ryan', 'rating':5},

            {'title':'Enemy at the Gates', 'rating':5},

            {'title':'Goodfellas', 'rating':4.5},

            {'title':'The Green Mile', 'rating':4.5},

            {'title':'Lord of the Rings', 'rating':3},            

            {'title':'Inception', 'rating':4},

            {'title':'Fight Club', 'rating':4.5},

            {'title':'Troy', 'rating':4.5},

            {'title':'Life Is Beautiful', 'rating':4},

            {'title':'Se7en', 'rating':4},

            {'title':'The Pianist', 'rating':5},           

            {'title':'The Usual Suspects', 'rating':4.5},

            {'title':'Léon: The Professional', 'rating':4},

            {'title':'Modern Times', 'rating':5},

            {'title':"City Lights", 'rating':4.5},

            {'title':'Dangal', 'rating':4.5},

            {'title':'Django Unchained', 'rating':4.5},

            {'title':'3 Idiots', 'rating':5},

            {'title':"Andhadhun", 'rating':4.5},

            {'title':'Inglourious Basterds', 'rating':3.5},

            {'title':'Children of Heaven', 'rating':5},        

            {'title':'Rang De Basanti', 'rating':4.5},

            {'title':'Indiana Jones and the Last Crusade', 'rating':4},

            {'title':'PK', 'rating':5},

            {'title':'Catch Me If You Can', 'rating':5},

            {'title':'The Terminator', 'rating':4},           

            {'title':'Hostel', 'rating':1.5},

            {'title':"Final Destination", 'rating':2},

            {'title':'Mirrors', 'rating':2},

            {'title':'Grudge, The', 'rating':2},

            {'title':'Lawrence of Arabia', 'rating':4},

            {'title':'Kung Fu Panda', 'rating':3},

            {'title':'Moana', 'rating':4}

         ] 

inputMovies = pd.DataFrame(userInput_Ferdous)

inputMovies.shape
#Filtering out the movies by title

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

#Then merging it so we can get the movieId. It's implicitly merging it by title.

inputMovies = pd.merge(inputId, inputMovies)

#Dropping information we won't use from the input dataframe

inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

#Final input dataframe

#If a movie you added in above isn't here, then it might not be in the original 

#dataframe or it might spelled differently, please check capitalisation.

inputMovies.shape
#Filtering out the movies from the input

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies.shape
#Resetting the index to avoid future issues

userMovies = userMovies.reset_index(drop=True)

#Dropping unnecessary issues due to save memory and to avoid issues

userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userGenreTable
inputMovies['rating'].shape
#Dot produt to get weights

userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

#The user profile

userProfile
#Now let's get the genres of every movie in our original dataframe

genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

#And drop the unnecessary information

genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

genreTable.head()
userProfile.sum()
#Multiply the genres by the weights and then take the weighted average

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

recommendationTable_df.head()
#Sort our recommendations in descending order

recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

#Just a peek at the values

recommendationTable_df.head()
#The final recommendation table

recommendation=movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(10).keys())]

recommendation
recommendation.to_csv("/kaggle/working/recommendation_content_based.csv")