# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Dataframe manipulation library

import pandas as pd

#Math functions, we'll only need the sqrt function so let's import only that

from math import sqrt

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Storing the movie information into a pandas dataframe

movies_df = pd.read_csv("../input/movielens-20m-dataset/movie.csv")

#Storing the user information into a pandas dataframe

ratings_df = pd.read_csv("../input/movielens-20m-dataset/rating.csv")
movies_df.shape
movies_df.head()
#Using regular expressions to find a year stored between parentheses

#We specify the parantheses so we don't conflict with movies that have years in their titles

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

#Removing the parentheses

movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)

#Removing the years from the 'title' column

movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

#Applying the strip function to get rid of any ending whitespace characters that may have appeared

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df.head()
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
userInput = [

            {'title':'Breakfast Club, The', 'rating':5},

            {'title':'Toy Story', 'rating':3.5},

            {'title':'Jumanji', 'rating':2},

            {'title':"Pulp Fiction", 'rating':5},

            {'title':'Akira', 'rating':4.5}

         ] 

inputMovies = pd.DataFrame(userInput)

inputMovies
#Filtering out the movies by title

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

#Then merging it so we can get the movieId. It's implicitly merging it by title.

inputMovies = pd.merge(inputId, inputMovies)

#Dropping information we won't use from the input dataframe

inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

#Final input dataframe

#If a movie you added in above isn't here, then it might not be in the original 

#dataframe or it might spelled differently, please check capitalisation.

inputMovies
#Filtering out the movies from the input

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies
#Resetting the index to avoid future issues

userMovies = userMovies.reset_index(drop=True)

#Dropping unnecessary issues due to save memory and to avoid issues

userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userGenreTable
inputMovies['rating']
#Dot produt to get weights

userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

#The user profile

userProfile
#Now let's get the genres of every movie in our original dataframe

genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

#And drop the unnecessary information

genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

genreTable.head()
genreTable.shape
#Multiply the genres by the weights and then take the weighted average

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

recommendationTable_df.head()
#Sort our recommendations in descending order

recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

#Just a peek at the values

recommendationTable_df.head()
#The final recommendation table

movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
movies_df.head()
#Dropping the genres column

movies_df = movies_df.drop('genres', 1)

movies_df.head()
ratings_df.head()
#Drop removes a specified row or column from a dataframe

ratings_df = ratings_df.drop('timestamp', 1)

ratings_df.head()
userInput = [

            {'title':'Breakfast Club, The', 'rating':5},

            {'title':'Toy Story', 'rating':3.5},

            {'title':'Jumanji', 'rating':2},

            {'title':"Pulp Fiction", 'rating':5},

            {'title':'Akira', 'rating':4.5}

         ] 

inputMovies = pd.DataFrame(userInput)

inputMovies
#Filtering out the movies by title

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

#Then merging it so we can get the movieId. It's implicitly merging it by title.

inputMovies = pd.merge(inputId, inputMovies)

#Dropping information we won't use from the input dataframe

inputMovies = inputMovies.drop('year', 1)

#Final input dataframe

#If a movie you added in above isn't here, then it might not be in the original 

#dataframe or it might spelled differently, please check capitalisation.

inputMovies
#Filtering out users that have watched movies that the input has watched and storing it

userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

userSubset.head()
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter

userSubsetGroup = userSubset.groupby(['userId'])
#lets look at one of the users, e.g. the one with userID=1130

userSubsetGroup.get_group(1130)
userSubsetGroup.head()
#Sorting it so users with movie most in common with the input will have priority

userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
# Check the first 3 users

userSubsetGroup[0:3]
# we only use a subset of top 100 users

userSubsetGroup = userSubsetGroup[0:100]
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient

pearsonCorrelationDict = {}



#For every user group in our subset

for name, group in userSubsetGroup:

    #Let's start by sorting the input and current user group so the values aren't mixed up later on

    group = group.sort_values(by='movieId')

    inputMovies = inputMovies.sort_values(by='movieId')

    #Get the N for the formula

    nRatings = len(group)

    #Get the review scores for the movies that they both have in common

    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]

    #And then store them in a temporary buffer variable in a list format to facilitate future calculations

    tempRatingList = temp_df['rating'].tolist()

    #Let's also put the current user group reviews in a list format

    tempGroupList = group['rating'].tolist()

    #Now let's calculate the pearson correlation between two users, so called, x and y

    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)

    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)

    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)

    

    #If the denominator is different than zero, then divide, else, 0 correlation.

    if Sxx != 0 and Syy != 0:

        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)

    else:

        pearsonCorrelationDict[name] = 0
pearsonCorrelationDict.items()
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')

pearsonDF.columns = ['similarityIndex']

pearsonDF['userId'] = pearsonDF.index

pearsonDF.index = range(len(pearsonDF))

pearsonDF.head()
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

topUsers.head()
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

topUsersRating.head()
#Multiplies the similarity by the user's ratings

topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']

topUsersRating.head()
#Applies a sum to the topUsers after grouping it up by userId

tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]

tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']

tempTopUsersRating.head()
#Creates an empty dataframe

recommendation_df = pd.DataFrame()

#Now we take the weighted average

recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']

recommendation_df['movieId'] = tempTopUsersRating.index

recommendation_df.head()
#sort it and see the top 20 recommendations



recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

recommendation_df.head(10)
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]