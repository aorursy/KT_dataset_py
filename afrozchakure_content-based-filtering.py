# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ast import literal_eval

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
md = pd.read_csv('../input/movies_metadata.csv')

md.head()
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# We are going to use much more suggestive metadata than Overview and Tagline.



links_small = pd.read_csv('../input/links_small.csv')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
links_small.head()
# Define a convert_int() function 

def convert_int(x):

    try:

        return int(x)

    except:

        return np.nan
# Convert the id's in the id column to interger

md['id'] = md['id'].apply(convert_int)

md[md['id'].isnull()]
# Removing the rows that have null values in the id column

md = md.drop([19730, 29503, 35587])
# Declaring the 'id' column as integer

md['id'] = md['id'].astype('int')
# We have 9099 movies avaiable in our small movies 

# metadata dataset which is 5 times smaller than our 

# original dataset of 45000 movies.

smd = md[md['id'].isin(links_small)]

smd.shape
# filling missing values in tagline and description columns

smd['tagline'] = smd['tagline'].fillna('')  # fill the empty data points in tagline column

smd['description'] = smd['overview'] + smd['tagline']  # Combining columns overview and tagline

smd['description'] = smd['description'].fillna('')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from surprise import Reader, Dataset, SVD, evaluate
# Converts a collection of raw documents to a matrix of TF-IDF features

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df = 0, stop_words='english')

tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape
# I will be using the Cosine Similarity to calculate a numeric quantity 

# that denotes the similarity between two movies. 

# Since we have used the TF-IDF Vectorizer, calculating the Dot Product 

# will directly give us the Cosine Similarity Score. Therefore, we will use 

# sklearn's linear_kernel instead of cosine_similarities since it is much faster.



cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
# We now have a pairwise cosine similarity matrix for all the movies in our dataset. 

# The next step is to write a function that returns the 30 most similar movies based 

# on the cosine similarity score.



smd = smd.reset_index()

titles = smd['title']  # Defining a new variable title

indices = pd.Series(smd.index, index = smd['title'])  # Defining a new dataframe indices
smd.head()
# Defining a function that returns 30 most similar movied bases on the cosine 

# similarity score

def get_recommendations(title):

    idx = indices[title]  # Defining a variable with indices

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    sim_scores = sim_scores[1: 31]  # Taking the 30 most similar movies

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]  # returns the title based on movie indices
# Getting the top recommendations for a few movies 

get_recommendations('Transformers').head(10)
# Our system is able to identify it as a Batman film and subsequently recommend 

# other Batman films as its top recommendations.

get_recommendations('The Dark Knight').head(10)
credits = pd.read_csv('../input/credits.csv')

keywords = pd.read_csv('../input/keywords.csv')
# Converting the keywords's id column to integer

keywords['id'] = keywords['id'].astype('int')

credits['id'] = credits['id'].astype('int')

md['id'] = md['id'].astype('int')
md.shape
# Merging teh credits and keywords DataFrame with md DataFrame

md = md.merge(credits, on = 'id')

md = md.merge(keywords, on = 'id')
# Creating a small movies DataFrame from the links_small

smd = md[md['id'].isin(links_small)]

smd.shape
# We now have our cast, crew, genres and credits, all in one dataframe.

from ast import literal_eval



smd['cast'] = smd['cast'].apply(literal_eval)

smd['crew'] = smd['crew'].apply(literal_eval)

smd['keywords'] = smd['keywords'].apply(literal_eval)

smd['cast_size'] = smd['cast'].apply(lambda x: len(x))  # Storing the cast_size

smd['crew_size'] = smd['crew'].apply(lambda x: len(x))  # Storing the crew_size
# Defining a function that gets the director's name 

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
# Applying the get_director function to the crew column to create smd['director'] column

smd['director'] = smd['crew'].apply(get_director)
# Arbitrarily we will choose the top 3 actors that appear in the credits list. 

smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)  # Taking the top 3 actor from cast
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# Strip Spaces and Convert to Lowercase from all our features. 

smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
# Mention Director 3 times to give it more weight relative to the entire cast.

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

smd['director'] = smd['director'].apply(lambda x: [x, x, x])
# Calculating the frequent counts of every keyword that appears in the dataset

s = smd.apply(lambda x: pd.Series(x['keywords']), axis = 1).stack().reset_index(level = 1, drop = True)

s.name = 'keywords'
s = s.value_counts()

s[: 5]
# Removing keywords that occur only once

s = s[s > 1]
# We will convert every word to its stem so that words such as Dogs and Dog 

# are considered the same.

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

stemmer.stem('dogs')
# Defining function filter_keywords() to create a list of words

def filter_keywords(x):

    words = []

    for i in x:

        if i in s:

            words.append(i)

    return words
# Converting the entries in keywords into stemmed words

smd['keywords'] = smd['keywords'].apply(filter_keywords)  #Applying the filter_keywords() function to keywords column

smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])  # Converting the entries in keywords column into stemmed words

smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])  # Removing spaces and converting the entries into lower case
smd.head()
# Combining 'keywords', 'cast', 'director' and genres columns in 'soup'

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']

smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
# Convert a collection of text documents to a matrix of token counts

count = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')

count_matrix = count.fit_transform(smd['soup'])
# Compute cosine similarity between samples in X and Y.

cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()  # Reseting the indices within smd

titles = smd['title']  # Storing smd['title'] in titles variable

indices = pd.Series(smd.index, index = smd['title'])    # Creating a DataFrame of indices
get_recommendations('The Dark Knight').head(10)
get_recommendations('Jurassic Park').head(10)
def improved_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    sim_score = sim_scores[1:26]

    movie_indices = [i[0] for i in sim_scores]

    

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]

    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')

    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()

    m = vote_counts.quantile(0.60)

    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]

    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]

    qualified['vote_count'] = qualified['vote_count'].astype('int')

    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)

    qualified = qualified.sort_values('wr', ascending=False).head(10)

    return qualified
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()

C
# The minimum votes required to be listed in the chart. We will use 95th percentile as our cutoff.

# i.e. For a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.

m = vote_counts.quantile(0.95)

m
# Defining the weighted_rating() function

def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)
# Getting recommedations from our model

improved_recommendations('Jurassic Park')