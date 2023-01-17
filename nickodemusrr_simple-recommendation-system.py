# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# importing data visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline



# Other packages will be imported later as we need it.



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dfGames = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

dfGames.head()
print(dfGames.columns.values)

print(dfGames.info())
dfGames[dfGames['Genre'].isnull()]
dfGames = dfGames.dropna(subset = ['Genre'])

dfGames.isnull().sum()
dfGames = dfGames[['Name', 'Platform', 'Genre']]
from sklearn.feature_extraction.text import CountVectorizer

model = CountVectorizer()

matrixGenre = model.fit_transform(dfGame['Genre'])

nameGenre = model.get_feature_names()

eventGenre = matrixGenre.toarray()



print(nameGenre)

print(eventGenre)
from sklearn.metrics.pairwise import cosine_similarity

score = cosine_similarity(matrixGenre)

print(score)
liked_game = 'Tekken 3'

liked_game_index = dfGames[dfGames['Name']==liked_game].index.values[0]



similarity_score = list(enumerate(score[liked_game_index]))

similar_game = sorted(similarity_score, key = lambda i : i[1], reverse=True)



# show 5 similar games

for i in similar_game[:5]:

    print(dfGames.iloc[i[0]]['Name'])
similar_game_clean = []

for i in similar_game:

    if i[1] > 0.5:

        similar_game_clean.append(i)



# show 5 similar games randomly

import random

recommendation = random.choices(similar_game_clean, k=5)



for i in recommendation:

    print(dfGames.iloc[i[0]]['Name'],

         dfGames.iloc[i[0]]['Platform'],

         dfGames.iloc[i[0]]['Genre'])