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
# Recommendation Systems

# Building a Non Personalized Recommendation Engine
# importing the libraries

import csv

import numpy as np

import pandas as pd
# building the critics data set

input_file = pd.read_csv("../input/movieratings.csv")
# Mean Rating

def topMean(prefs,n=5):

    #calculating the mean and sorting the values from highest

    scores=prefs.mean().sort_values(ascending=False)

    return scores[0:n]
topMean(input_file, n=5)
# % of ratings r+: Calculating the percentage of ratings for each movie 

# that are r or higher.



def topPerc(prefs,r=3,n=5):

    #turning movie  titles to a column to handle it easier 

    input_titles=pd.melt(prefs, id_vars=["User"], var_name='title')

    #I filtered for values higher than r

    high_rank=input_titles.loc[input_titles['value'] >= r]

    #I counted the number of reviews for ratings 3,4,5(or any arbitrary ratings this is for our default case) grouped by title, and summed them to get the aggregate number of reviews higher than r

    rank=high_rank.groupby(['title','value']).count().unstack().sum(axis=1)

    #the same procedure but for all ratings irrespective of our threshhold r to get the percentage later

    total=input_titles.groupby(['title','value']).count().unstack().sum(axis=1)

    #here I used the total number of rankings higher than r divided by the total number of rankings, and sorted the values

    scores=(rank/total).sort_values(ascending=False)

    return scores[0:n]
topPerc(input_file)
# Rating Count: Counting the number of ratings for each movie, order with 

# the most number of ratings first, and submit the top n.



def topCount(prefs,n=5):

    #I just took the numeric values of the my data, which are the ratings, counted them and sorted

    scores=prefs.count(numeric_only=True).sort_values(ascending=False)

    return scores[0:n]
topCount(input_file)
# Top 5 Movies related: Calculating movies that most often occur with 

# other movie, Star Wars: Episode IV - A New Hope (1977) by defautl

# using the (x+y)/x method. In other words, for each movie, calculate the 

# percentage of the other movie raters who also rated that movie. Order with 

# the highest percentage first, and submit the top 5.



def topOccur(prefs,x='260: Star Wars: Episode IV - A New Hope (1977)',n=5):

    #Here I am dropping the NA values because we need to take the users that bought/watched this one movie we select as our item #1

    prefs = prefs[prefs[x].notna()]

    #we count them

    items=len(prefs[x].notna())

    #I need two empty lists where I will be storing the number of ratings(NAs excluded) and the title of the movie

    my_movie=[]

    otro_movies=[]

    #starting a loop that takes all the columns of our prefs(input_file), neglecting the first column, which is User names

    for movie in prefs.columns[1:input_file.shape[0]]:

        #also we need to make sure the loop is not comparing the person who watched the movie to themselves, so the condition is that it's not the same movie

        if (movie != x):

            #again dropping NAs and storing the movie titles that rated both

            test=prefs[prefs[movie].notna()]

            #appending the lists one to store the length(basically count) for the movie titles we had

            my_movie.append(len(test[movie]))

            #the other just all so that we can calculate the percentage

            otro_movies.append(movie)

    #to handle it easier we are creating a dataframe where we have count of ratings and movie title        

    test=pd.DataFrame(my_movie,otro_movies)

    #now we just calculate percentage, and sort them from highest

    scores=(test.iloc[:,0]/items).sort_values(ascending=False)

    return scores[0:n]

topOccur(input_file,x='260: Star Wars: Episode IV - A New Hope (1977)',n=5)