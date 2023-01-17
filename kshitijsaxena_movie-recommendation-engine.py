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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from surprise import Reader, Dataset, SVD, evaluate



import warnings; warnings.simplefilter('ignore')
metadata = pd.read_csv('../input/movies_metadata.csv')

metadata.head()
metadata['genres'] = metadata['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = metadata[metadata['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = metadata[metadata['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()

C
# Assumption:

# Top 5% vote counts are reliable, others are very small to analyse

m = vote_counts.quantile(0.95)

m
metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = metadata[(metadata['vote_count'] >= m) & (metadata['vote_count'].notnull()) & (metadata['vote_average'].notnull())]

qualified = qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int')

qualified['vote_average'] = qualified['vote_average'].astype('int')

qualified.shape
def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
qualified.head(15)
s = metadata.apply(lambda x: pd.Series(x['genres']) , axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'

gen_md = metadata.drop('genres', axis=1).join(s)
def build_chart(genre, percentile=0.85):

    df = gen_md[gen_md['genre'] == genre]

    vote_count = df[df['vote_count'].notnull()]['vote_count'].astype('int')

    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()

    m = vote_count.quantile(percentile)

    

    qualified = df[(df['vote_count'] >= m) 

                   & (df['vote_count'].notnull()) 

                   & (df['vote_average'].notnull())][[

        'title', 'year', 'vote_count', 'vote_average', 'popularity'

    ]]

    qualified['vote_count'] = qualified['vote_count'].astype('int')

    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(

        lambda x: x['vote_count']/(x['vote_count'] + m) * x['vote_average'] 

        + (m/(m+x['vote_count']) * C), axis=1)

    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified
build_chart('Horror').head(15)
links_small = pd.read_csv('../input/links_small.csv')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
metadata = metadata.drop([19730, 29503, 35587])

# Droping these rows because ids are not parcable as int

# The data is not clean

metadata['id'] = metadata['id'].astype('int')
smd = metadata[metadata['id'].isin(links_small)]

smd.shape
smd['tagline'] = smd['tagline'].fillna('')

smd['description'] = smd['overview'] + smd['tagline']

smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(smd['description'])

tfidf_matrix.shape
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim[0]
smd = smd.reset_index()

titles = smd['title']

indices = pd.Series(smd.index, index=smd['title'])
def get_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]
get_recommendations('The Dark Knight Rises').head(25)
credits = pd.read_csv('../input/credits.csv')

keywords = pd.read_csv('../input/keywords.csv')
keywords['id'] = keywords['id'].astype('int')

credits['id'] = credits['id'].astype('int')

metadata['id'] = metadata['id'].astype('int')
metadata.shape
metadata = metadata.merge(credits, on='id')

metadata = metadata.merge(keywords, on='id')
smd = metadata[metadata['id'].isin(links_small)]

smd.shape
smd['cast'] = smd['cast'].apply(literal_eval)

smd['crew'] = smd['crew'].apply(literal_eval)

smd['keywords'] = smd['keywords'].apply(literal_eval)

smd['cast_size'] = smd['cast'].apply(lambda x: len(x))

smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
def get_director(x):

    for i in x:

        if i['job'].lower() == 'director':

            return i['name']

        return np.nan
smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [i.lower().replace(" ", "") for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

smd['director'] = smd['director'].apply(lambda x: [x, x, x])
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'keyword'
s = s.value_counts()

s[:10]
s = s[s > 1]
stemmer = SnowballStemmer('english')

stemmer.stem('Dogs')
def filter_keywords(keywords):

    filtered_keywords = []

    for word in keywords:

        if word in s:

            filtered_keywords.append(word)

    return filtered_keywords
smd['keywords'] = smd['keywords'].apply(filter_keywords) 
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])

smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['keywords'].head()
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']

smd['soup'].head()
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()

titles = smd['title']

indices = pd.Series(smd.index, index=smd['title'])
get_recommendations('The Dark Knight').head(10)
get_recommendations('Thursday').head(10)
def improved_reco(title, percentile=0.60, num_recommendations=10):

    idx = indices[title]

    if type(idx) == pd.Series:

        sim_scores = []

        for i in idx:

            sim_scores = sim_scores + list(enumerate(cosine_sim[i]))

    else:

        sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:26]

    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'popularity', 'year']]

    

    vote_count = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')

    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

    m  = vote_count.mean()

    C = vote_averages.quantile(percentile)

    qualified = movies[(movies['vote_count'].notnull()) 

                       & (movies['vote_average'].notnull()) 

                       & (movies['vote_count'] >= m)]

    qualified['vote_count'] = qualified['vote_count'].astype('int')

    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)

    qualified = qualified.sort_values('wr', ascending=False).head(num_recommendations)

    return qualified
improved_reco('Kahaani')