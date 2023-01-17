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

userSubsetGroup.get_group(1130)
#Sorting it so users with movie most in common with the input will have priority

userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup[0:3]
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
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

recommendation_df.head(10)
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]