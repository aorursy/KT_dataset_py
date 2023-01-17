# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
movie = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')
rating= pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')
movie.head()
rating.head()
movie['year'] = movie.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movie['year'] = movie.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movie['title'] = movie['title'].apply(lambda x: x.strip())
movie.head()
movie.drop(columns=['genres'], inplace=True)
movie.head()
rating.drop(columns=['timestamp'],inplace=True)
rating.head()
user = [
            {'title':'Breakfast Club, The', 'rating':4},
            {'title':'Toy Story', 'rating':2.5},
            {'title':'Jumanji', 'rating':3},
            {'title':"Pulp Fiction", 'rating':4.5},
            {'title':'Akira', 'rating':5}
         ] 
inputMovie = pd.DataFrame(user)
inputMovie
#Filtering out the movies by title
Id = movie[movie['title'].isin(inputMovie['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovie = pd.merge(Id, inputMovie)
#Dropping information we won't use from the input dataframe
inputMovie = inputMovie.drop('year', 1)
inputMovie
#Filtering out users that have watched movies that the input has watched and storing it
users = rating[rating['movieId'].isin(inputMovie['movieId'].tolist())]
users.head()
users.shape
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = users.groupby(['userId'])
#showing one such group example by getting all the users of a particular uderId
userSubsetGroup.get_group(1130)
#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup[0:3]
userSubsetGroup = userSubsetGroup[0:100]
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovie = inputMovie.sort_values(by='movieId')
    #Get the N for the formula
    n = len(group)
    #Get the review scores for the movies that they both have in common
    temp = inputMovie[inputMovie['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp['rating'].tolist()
    #put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(n)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(n)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(n)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorDict[name] = 0

pearsonCorDict.items()
pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
topUsersRating=topUsers.merge(rating, left_on='userId', right_on='userId', how='inner')
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
movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
