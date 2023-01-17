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
import statistics as st
input_file=pd.read_csv('../input/movie-ratings/movieratings.csv',index_col=0, sep=",")
#defining column names and reading the text file to get a dataframe
rating_columns = ['User', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data', sep='\t', names=rating_columns)

movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep='|', names=movie_columns, usecols=range(5),encoding='latin-1')

#as we didn't have all the info in one file, we are merging two files to get movie names, user ids and ratings
movie_ratings = pd.merge(movies, ratings)

#for further manipulations we will need only these three columns and we need the index to be the user to be able to manipulate data easier
movie_ratings=movie_ratings.loc[:,['title','User','rating']].set_index('User')

#we also need to redesign the table to match the look of our smaller dataset so that it can be manipulated the same way, this structure is not randomly chosen, it fits the best manipulations with iterations and dataframe calculations
data_big=movie_ratings.reset_index().groupby(['User', 'title'])['rating'].aggregate('first').unstack()

#as opposed to smaller dataset we don't have names here, hence we transform the index to strings to have names of people rather than numbers even if it's only ids
data_big.index=data_big.index.map(str)
# Recommendation algorithms

# Pearson correlation coefficient for person1 and person2

def pearsonSimilarity(prefs,person1,person2):
#dropping all the columns that are not mutually shared between two people
    df_1=prefs.loc[[person1, person2],:].dropna(axis='columns')
#finding correlation between the ratings that are left aka for the same movies both people watched
    scores=df_1.loc[person1].corr(df_1.loc[person2])
    return scores
pearsonSimilarity(data_big,'308','287')
# Geting recommendations for person by using a weighted average
# of every other user's rankings

def getRecommendations(prefs,person,similarity=pearsonSimilarity):
#I am using np seterr as there are some NaNs and 0 values which might occur in the calculations only for the big dataset, which can be ignored and no how affect the end results, if it was crucial, certianly I would not have handled a warning by ignoring, but in Recommender we truly don't care about 0 values, which don't provide any valuable insight
    np.seterr(divide='ignore', invalid='ignore')
#extracting the movies that the person watched in order to take it out later from the recommendations
    person_watched=pd.DataFrame(prefs.loc[person,:].dropna()).drop(columns=[person])
    movieratings=prefs
    for title in person_watched.index:
        movieratings=movieratings.drop(columns=title)
    list_1=[]
    list_2=[] 
#iterating through the movies that the person did not watch and adding correlations for all other users, excluding the person we are making the recommendations for
    for i in movieratings.index:
         if i != person:
                list_1=(i,pearsonSimilarity(prefs,person,i))
                list_2.append(list_1)    
    list_2=pd.DataFrame(list_2)
    list_2.columns=['User','Similarity']
    
#merging the correlations with ratings dataframe to perform calculations
    main_data = pd.DataFrame.merge(list_2,movieratings, on='User',left_on=None, right_on=None)
#multiplying similiarity with rating for every user and movie
    for col in main_data.columns[2:]:
         main_data[col] = np.where(main_data.loc[:,col]=="NaN",main_data.loc[:,col],main_data.loc[:,'Similarity']*main_data.loc[:,col])

#because we need to handle the negative values in the correlation, I set a threshhold of 0 and cliped it, while also filling NAs with 0, this was the most efficient way not to distort the calculations, because in the formula we have sums, zeros would not cause any issues
    main_data.iloc[:,1:]=main_data.iloc[:,1:].astype("float").clip(0).fillna(0)
    rankings=[]
#here we needed to take into account similiarities only for the movies-wise, meaning if for movie X the user did not have a rating(or as we filled them the rating was 0), we need to skip taking their similiarities into account. Hence, I used the trick we did during our R class to have an extra column that allows us to manipulate the data easier. 
    for u in main_data.columns:
        if (u != 'User') and (u!='Similarity'):
            main_data.loc[main_data[u] !=0, 'Indicator'] = 1
            main_data.loc[main_data[u] ==0, 'Indicator'] = 0
#we create an if in case of movies that have 0 as a result, if there was no 'if' we would get an error that we can't divide by zero, hence, we need to take into account this specific cases
            if sum(main_data['Similarity'].mul(main_data['Indicator'])) == 0:
                r=0
            else:
                r=(sum(main_data[u])/sum(main_data['Similarity'].mul(main_data['Indicator'])))
                if r>5:
                    r=5
#printing the rating and movie name and appending in the list
            list_3=(r,u)
            rankings.append(list_3)
#I am applying the sorted function on the tuples and sorting by the rankings value from highest to lowest
            rankings = sorted(rankings, key=lambda tup: tup[0],reverse=True)
    return rankings

getRecommendations(data_big,'308')
# Returning the best matches for person from the prefs dictionary. 
# Number of results and similarity function are optional params.

def topMatches(prefs,person,n=5,similarity=pearsonSimilarity):
#initializing lists where the results will be registered
    list_1=[]
    list_2=[]
#basically this 'for loop' goes through all the people in our dataframe, excluding the person taken, pulls out the similiarities and if they are not NaNs then it appends them to the list, in this case we get NaNs when people don't have anything to do with each other hence their correlation is registered as NaN
    for i in prefs.index:
        if i != person:
            list_1=(pearsonSimilarity(prefs,person,i),i)
            if np.isnan(list_1[0])==False:
                list_2.append(list_1)
#here I am applying the sorted function on the tuples and sorting by the score value from highest to lowest, top 5
        scores = sorted(list_2, key=lambda tup: tup[0],reverse=True)
    return scores[0:n]
topMatches(data_big,'308')