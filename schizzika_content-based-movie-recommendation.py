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
text = ["London Paris London", "Paris Paris London"]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

print(cv.get_feature_names())
print(count_matrix.toarray())
from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(count_matrix)

print(similarity_scores)
def get_title_from_index(index):

    return df[df.index == index]['title'].values[0]
def get_index_from_title(title):

    return df[df.title == title]['index'].values[0]
df = pd.read_csv("/kaggle/input/code-heroku/movie_dataset.csv")
df
df.columns
df.fillna('', inplace = True)
features = ['keywords', 'cast', 'genres', 'director']
def combine_features(row):

    return row['keywords'] + " " + row['genres'] + " " + row['cast'] + " " + row["director"]
df["combined_features"] = df.apply(combine_features, axis = 1)

print("Combined Features: ", df["combined_features"].head())
count_matrix = cv.fit_transform(df['combined_features'])
cosine_similarity = cosine_similarity(count_matrix)
print(cosine_similarity)
movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_similarity[movie_index]))
sorted_similar_movies = sorted(similar_movies, key = lambda x: x[1], reverse = True)
counter = 0

for movie in sorted_similar_movies:

    print(get_title_from_index(movie[0]))

    counter = counter + 1

    if counter > 50:

        break