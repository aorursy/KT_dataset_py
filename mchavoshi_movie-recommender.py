import pandas as pd

from math import sqrt

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
movies_df = pd.read_csv('/kaggle/input/grouplens-2018/ml-latest/movies.csv')

ratings_df = pd.read_csv('/kaggle/input/grouplens-2018/ml-latest/ratings.csv')
movies_df.shape
movies_df.tail()
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
#Dropping the genres column, no need for them

movies_df = movies_df.drop('genres', 1)

movies_df.head()
ratings_df.head()
#Drop removes a specified row or column from a dataframe

ratings_df = ratings_df.drop('timestamp', 1)

ratings_df.head()
# here's a hypothetical user that we want to make suggestions for

userInput = [

            {'title':'Avatar 2', 'rating':7},

            {'title':'13 Hours', 'rating':3.5},

            {'title':'Jumanji', 'rating':7},

            {'title':"Sherlock: The Abominable Bride", 'rating':8},

            {'title':'Jurassic World', 'rating':8},

    {'title':'Star Wars: Episode VII - The Force Awakens', 'rating':6},

    {'title':'Avengers: Age of Ultron', 'rating':9},

    {'title':'Ant-Man', 'rating':8},

    {'title':'Justice League: Throne of Atlantis', 'rating':7}]

inputMovies = pd.DataFrame(userInput)

inputMovies
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

inputId.head()
inputMovies = pd.merge(inputId, inputMovies)

inputMovies
inputMovies = inputMovies.drop('year', 1)

inputMovies
#Filtering out users that have watched movies that the input has watched and storing it

userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

userSubset.head()
userSubsetGroup = userSubset.groupby(['userId'])
#Sorting it so users with movie most in common with the input will have priority

userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup[0]
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

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')

pearsonDF.columns = ['similarityIndex']

pearsonDF['userId'] = pearsonDF.index

pearsonDF.index = range(len(pearsonDF))

pearsonDF.head()
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

topUsers.head()
topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

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