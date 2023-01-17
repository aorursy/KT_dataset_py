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
credits = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")



credits.head()
movies.head()
print(credits.shape)

print(movies.shape)
credits = credits.rename(index=str, columns={"movie_id": "id"})

credits.head()
df_merge = movies.merge(credits, on="id")

df_merge.head()
df_cleaned = df_merge.drop(columns=["homepage", "title_x", "title_y", "status", "production_countries"])

df_cleaned.head()
df_cleaned.describe()
df_cleaned.info()
df_cleaned['overview'].head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer(

    min_df = 3,

    max_features = None,

    strip_accents = "unicode",

    analyzer = "word",

    token_pattern = r'\w{1,}',

    ngram_range = (1, 3),         # Taking combinations of 1-3 different kind of words

    stop_words = "english"        # Remove the unnecessary stopword characters

)



df_cleaned['overview'] = df_cleaned['overview'].fillna('')   # Removing NaN values



tfv_matrix = tfv.fit_transform(df_cleaned['overview'])   # => Sparse Matrix(vectors) => most of the values in matrix = 0

tfv_matrix
tfv_matrix.shape

# 4803 records  and 10417 => features(based on the combinations of words(ngram=(1, 3)))
from sklearn.metrics.pairwise import sigmoid_kernel

# Sigmoid => Responsible for transforming input between 0 to 1

# Passing the summary vectors in the sigmoid function => Will get values between 0 and 1



sig = sigmoid_kernel(tfv_matrix, tfv_matrix)    # Combination of the same matrix

sig[0]

# Overview 1 related to overview 1, overview 1 related to overview 2, overview 1 related to overview 3, and so on
# Mapping of Indices and Corresponding Movie Titles in the dataset

indices = pd.Series(df_cleaned.index, index=df_cleaned['original_title']).drop_duplicates()

indices
# Movie Index for the movie "Shanghai Calling"

print(indices['Shanghai Calling'])

print(sig[4801])
# Converting the range of the sigmoid values to a list along with respective indices using the enumerate function

# [(Index, Score)]

list(enumerate(sig[indices['Shanghai Calling']]))
# Sorting according to the scores in the list

sorted(list(enumerate(sig[indices['Shanghai Calling']])), key=lambda x: x[1], reverse=True)
# Give movie title as input and based on the movie title we apply the object created in the sigmoid kernal

# Doing the same thing as above but putting it in the fuction

def give_recommendation(title, sig=sig):

    idx = indices[title]

    sig_scores = list(enumerate(sig[idx]))

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    

    # Getting Top 10 scores(index, scores)

    sig_scores = sig_scores[1:11]

    movie_indices = [i[0] for i in sig_scores]

    return df_cleaned['original_title'].iloc[movie_indices]
give_recommendation('Shanghai Calling')
give_recommendation('Avatar')