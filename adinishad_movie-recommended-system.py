# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 25)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load dataset

movie = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

credit = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")
# first look

movie.head()
credit.head()
# merge two dataset

credit.columns = ['id','cast', 'title', 'crew']

movie= movie.merge(credit, on='id')
movie.head(3)
movie.info() # some information of dataset
# Wordcolud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
# word cloud function

def cloud(col):    

    wcloud = " ".join(f for f in movie[col])

    wc_ = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = STOPWORDS)

    wc_.generate(wcloud)

    plt.subplots(figsize=(10,6))

    plt.imshow(wc_, interpolation="bilinear")

    plt.axis("off")
# for title column

cloud("original_title")
# fill overview column

movie["overview"] = movie["overview"].fillna("")
cloud("overview")
# Tfidf Vectorize

from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(stop_words="english")



tfidf_matrix = tfidf.fit_transform(movie["overview"])



tfidf_matrix
# we will use sklearn's linear_kernel() instead of cosine_similarities() since it is faster.

from sklearn.metrics.pairwise import linear_kernel



cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# reverse map of indices and movie original_title

indices = pd.Series(movie.index, index=movie['original_title']).drop_duplicates()
# recommendation function



def get_recommendation(title, cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movies = [i[0] for i in sim_scores]

    movies = movie["original_title"].iloc[movies]

    return movies
# IF you search "Spectre", name of the movies below will be recommended

get_recommendation('Spectre', cosine_sim)
get_recommendation("John Carter", cosine_sim)
# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['keywords', 'genres']

for feature in features:

    movie[feature] = movie[feature].apply(literal_eval)
movie[['original_title', 'keywords', 'genres']].head(3)
# Extract list of genres

def list_genres(x):

    l = [d['name'] for d in x]

    return(l)

movie['genres'] = movie['genres'].apply(list_genres)



# Extract list of keywords

def list_keyword(y):

    i = [a['name'] for a in y]

    return(i)

movie['keywords'] = movie['keywords'].apply(list_keyword)
# join genre and keywords

def genre(x):

    return ''.join(' '.join(x['genres']) + ' ' + ' '.join(x['keywords']))



# new column

movie['mix'] = movie.apply(genre, axis=1)
movie["mix"]
# Countvectorizer

from sklearn.feature_extraction.text import CountVectorizer



countvect = CountVectorizer(stop_words="english")



countvect_mat = tfidf.fit_transform(movie["mix"])



countvect_mat
from sklearn.metrics.pairwise import cosine_similarity



cos_sim = cosine_similarity(countvect_mat, countvect_mat)
# reverse map of indices and movie original_title

movie = movie.reset_index()

indices = pd.Series(movie.index, index=movie['original_title'])
get_recommendation("John Carter", cos_sim)
get_recommendation("Soldier", cos_sim)
# avarage rating

avg = movie["vote_average"].mean()

#  We will use 90th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 90% of the movies in the list.

q = movie["vote_count"].quantile(0.9)
print(avg)

print(q)
movies = movie[movie["vote_count"] >= q]
# function of weighted_rating 

def weighted_rating(x, q=q, avg=avg):

    v = x['vote_count']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+q) * R) + (q/(q+v) * avg)
# apply for qualfied movies

movies["score"] = movies.apply(weighted_rating, axis=1)
# Sort movies based on score calculated above

movies = movies.sort_values('score', ascending=False)



# Print the top 10 movies

listed = movies[['original_title', 'vote_count', 'vote_average', 'score', "popularity"]].head(10)
# Visualize

import seaborn as sns





plt.subplots(figsize=(10,6))

sns.barplot(listed["score"], listed["original_title"], palette="Set2")

plt.title("Movie Vs Score")
popular= movies.sort_values('popularity', ascending=False)

plt.figure(figsize=(12,4))



plt.barh(popular['original_title'].head(10),popular['popularity'].head(10), align='center',

        color="#313131")

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")