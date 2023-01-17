# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv("../input/movie.csv")

ratings = pd.read_csv("../input/rating.csv")
movies
#take a sample of the dataset

ratings = ratings.iloc[:1000000]
ratings.info()
#there are several ways to combine datasets, merge is very flexible

joined = movies.merge(ratings,left_on = 'movieId',right_on='movieId')
joined
joined['genres'].str.split("|")
joined['genres_split'] = joined.genres.str.split("|")

joined
#we can iterate over a dataframe and perform operations, accessing each element separately



for i in joined.iterrows():

    print(i)

    break
%%time

#loop over a dataframe and create a list of each unique genre



unique_genres = []

for i in joined.iterrows():

    for element in i[1]['genres_split']:

        if element not in unique_genres:

            unique_genres.append(element)
%%time

#a faster way than loops

unique_g = []

joined['genres_split'].apply(lambda x: [unique_g.append(i) for i in x])

list(set(unique_g))
unique_genres
%%time

genres = []

for i in joined.iterrows():

    for g in i[1]['genres_split']:

        genres.append([i[1]['title'],g])
%%time

genres_fast = []

joined.apply(lambda x: [genres_fast.append([x['title'],i]) for i in x['genres_split']],axis=1)
genres_fast
genres_df = pd.DataFrame(genres,columns=['title','genre'])

genres_df
genres_df['value'] = 1
genres_df = genres_df.groupby(['title','genre']).mean()

genres_df.reset_index(inplace=True)

genres_df
genres_p = genres_df.pivot(columns='genre',index='title',values='value')

genres_p
prepared_df = joined.merge(genres_p,left_on='title',right_on='title')
prepared_df