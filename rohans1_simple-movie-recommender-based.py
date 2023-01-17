# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
url = '/kaggle/input/grammar-and-online-product-reviews/GrammarandProductReviews.csv'

df = pd.read_csv(url)

df.head()
#data overview

print('Rows: ', df.shape[0])

print('columns: ', df.shape[1])

print('\nfeatures: ', df.columns.tolist())

print('\nMissing values: ', df.isnull().values.sum())

print('\nUnique values: \n', df.nunique() )


df.info()#looking at the value types in the features

df.isnull().sum() #looking for sum of null values in each column
#finding the general statics of the dataset

df.describe()
#filtering the dataset based on movies and music both

df = df[df['categories'].str.contains('Movies') | df['categories'].str.contains('Music')]

df
#counting the no of unique categories

df['categories'].value_counts()
#counting values of unique names of products

df['name'].value_counts()
df.nunique()
df['reviews.username'].value_counts()
#10 popular movies(from 80 unique movies, 19449 users ) and plotting them

df['name'].value_counts()[0:10].plot('barh', figsize=[10,6], fontsize=20).invert_yaxis()
#new df with list of user and favourite moveis only

df_user_movie = df[['reviews.username', 'name', 'reviews.rating']]

df_user_movie =df_user_movie[df_user_movie['reviews.rating'] ==5]

df_user_movie = df_user_movie.drop(columns=['reviews.rating'], axis=1)

df_user_movie
#converting the dataframe to a cav file

df_user_movie.to_csv('df_user_movie.csv', index= False)
#reading the file

from collections import defaultdict

from collections import Counter

import csv

pd.read_csv('/kaggle/working/df_user_movie.csv').head(10)
user_movie_map = defaultdict(list)

movie_user_map = defaultdict(list)

with open('df_user_movie.csv', 'r') as csvfile:

    w = csv.reader(csvfile, delimiter=',')

    for row in w:

        user_movie_map[row[0]].append(row[1])

        movie_user_map[row[1]].append(row[0])
df_user_movie['reviews.username'].value_counts()
df_user_movie.nunique()
user_movie_map['Jess']
#all the moveis that  "Mike" liked:

user_movie_map['Mike']
#all the users that liked(movie as key):

movie_user_map["Chuggington: Let's Ride The Rails"]

#notice user "Michael" in there
def get_similar_movie(user_movie_map, movie_user_map, m):

    biglist = []

    for u in movie_user_map[m]: #get all user that liked that movie

        biglist.extend(user_movie_map[u])#finf all other moveis those user s liked and add to biglist. append list to list

    

    return Counter(biglist).most_common(11)[1:]#use counter to 'count' the other movies that show up most common

def get_movie_recommendation(user_movie_map,movie_user_map,u1):

    biglist = []

    for m in user_movie_map[u1]: # for the movies a specific user likes

        for u in movie_user_map[m]: # get other users who liked those movies

            biglist.extend(user_movie_map[u]) # find the other movies those "similar folks" most liked

    return Counter(biglist).most_common(10) # return tuples of (most common id, count)
get_similar_movie(user_movie_map, movie_user_map, "The Resident Evil Collection 5 Discs (blu-Ray)")  # movie
get_movie_recommendation(user_movie_map, movie_user_map, 'oldandwise')  # user
get_movie_recommendation(user_movie_map, movie_user_map, 'jess')