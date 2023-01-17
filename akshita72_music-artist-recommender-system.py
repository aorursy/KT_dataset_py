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
data = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_artist.csv')

print(data.columns)

data.head()
data.drop(['duration_ms','key','mode','count'],axis=1,inplace=True)
data.shape
data['popularity'] = data['popularity']/100

data['tempo'] = (data['tempo'] - 50)/100

data['loudness'] = (data['loudness'] + 60)/60
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 

            'loudness', 'speechiness', 'tempo', 'valence', 'popularity']

def createUserPrefMatrix(artistRatingDict):

    

    artists = artistRatingDict.keys()

    artMat = data[data['artists'].isin(artists)]

    #print(artMat)

    

    for artist, rating in artistRatingDict.items():

        artMat.loc[artMat['artists']==

                   artist,features] = artMat.loc[artMat['artists']==

                                                 artist,features].mul(rating,axis=0)

    

    userProfile = artMat.loc[:,features].sum(axis=0)

    normalized_userProfile = (userProfile/userProfile.sum())*10

    

    return normalized_userProfile



def createRecomMatrix(userProfile,artists):

    

    artMat = data[~data['artists'].isin(artists)]

    artMat.set_index('artists',inplace=True) 

    #print(userProfile)

    #print(artMat.head())

    

    recomMat = pd.DataFrame(artMat.values*userProfile.values, 

                            columns=artMat.columns, index=artMat.index)

    recomMat = recomMat.sum(axis=1)

    recomMat.sort_values(ascending = False,inplace=True)

    

    return recomMat



def recommend(artistRatingDict):

    

    userProfile = createUserPrefMatrix(artistRatingDict)

    

    recommendationMat = createRecomMatrix(userProfile,

                                          artistRatingDict.keys()) 

    

    return recommendationMat.head(10)
import random

artists = random.sample(list(data['artists']),k=10)

ratings = [10,10,8,5,9,2,3,7,6,10]

dictionary = dict(zip(artists, ratings))

print(dictionary)
recommend(dictionary)