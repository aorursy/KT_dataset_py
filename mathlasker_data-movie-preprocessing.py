import os

import json

import numpy as np

import pandas as pd

import seaborn as sns

import random

from gensim.models import Word2Vec 

import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm

from IPython.display import Image, display

from IPython.core.display import HTML

tqdm.pandas()

%matplotlib inline
files = os.listdir('/kaggle/input/tmdballmovies/movies')
len(files)
movie_data = []

with tqdm(total = len(files)) as pbar:

    for i in files:

        try:

            with open(f"/kaggle/input/tmdballmovies/movies/{i}", encoding="utf-8") as json_file:

                data = json.load(json_file)

                movie_data.append(data)

        except:

            print(i)

        pbar.update(1)

  
movies_df = pd.DataFrame(movie_data)
movies_df.head(3)
# sort the data frame by 'id'

movies_df = movies_df.iloc[movies_df['id'].sort_values(ascending = True).index]

movies_df.index = range(movies_df.shape[0])
movies_df.isnull().sum().sort_values(ascending= False)
(movies_df == '').sum().sort_values(ascending=False)
movies_df = movies_df.drop(index = movies_df[movies_df['adult'] == True].index.values)

movies_df.index = range(movies_df.shape[0])
movies_df = movies_df.drop(index = movies_df[movies_df['poster_path'].isnull()].index.values)

movies_df.index = range(movies_df.shape[0])
# drop columns that we arent go to use

drop_columns = ['adult', 'backdrop_path', 'belongs_to_collection', 'video']

movies_df = movies_df.drop(columns = drop_columns)

# Reset Index

movies_df.index = range(movies_df.shape[0])
movies_df.info()
movies_df = movies_df.drop(index = movies_df[movies_df['poster_path'] == ''].index.values)
movies_df.index = range(movies_df.shape[0])
# Get the movie's year

def get_release_year(release_date):

    if release_date != '':

        year_month_day = release_date.split('-')

        year = year_month_day[0]

        return int(year)

    else:

        return 0



# Get the movie's day

def get_release_day(release_date):

    if release_date != '':

        year_month_day = release_date.split('-')

        day = year_month_day[2]

        return int(day)

    else:

        return 0



# Get the movie's month

def get_release_month(release_date):

    if release_date != '':

        year_month_day = release_date.split('-')

        month = year_month_day[1]

        return int(month)

    else:

        return 0
# obtain release_year from release_date.

movies_df['release_year'] = movies_df['release_date'].progress_apply(get_release_year)

movies_df['release_day'] = movies_df['release_date'].progress_apply(get_release_day)

movies_df['release_month'] = movies_df['release_date'].progress_apply(get_release_month)
movies_df.head(3)
# show

movies_df[['release_year', 'release_month', 'release_day', 'release_date']].head(3)
# ['overview', 'tagline']

def get_bool_tagline(tagline):

    if tagline != '':

        return 1

    else:

        return 0

    

def fill_empty_tagline(tagline):

    if tagline == '':

        return ''

    else:

        return tagline

    

def get_bool_overview(overview):

    if overview != '':

        return 1

    else:

        return 0

    

def fill_empty_overview(overview):

    if overview == '':

        return ''

    else:

        return overview
# fill na

movies_df[['overview', 'tagline']] = movies_df[['overview', 'tagline']].fillna('')
movies_df['overview'] = movies_df['overview'].progress_apply(fill_empty_overview)

movies_df['bool_overview'] = movies_df['overview'].progress_apply(get_bool_overview)
movies_df['tagline'] = movies_df['tagline'].progress_apply(fill_empty_tagline)

movies_df['bool_tagline'] = movies_df['tagline'].progress_apply(get_bool_tagline)
def get_genres(list_genres):

    """receive the list of genres and return a string genre1,genre2,genre3 in orther to use similarity score first"""

    """False: 0, True: 1"""

    if len(list_genres) != 0:

        genres = ''

        for genre_dict in list_genres:

            genres = genres+ ',' + genre_dict['name']

        genres = genres[1:]

    else:

        genres = ''

    return genres



def get_bool_genres(list_genres):

    """receive a string of genres separated by comma and return if there is genres then is True, and if it unknown

    return False"""

    if list_genres != '':

        return 1

    else:

        return 0
movies_df['genres'] = movies_df['genres'].progress_apply(get_genres)

movies_df['bool_genre'] = movies_df['genres'].progress_apply(get_bool_genres)
def get_production_companies(list_companies):

    """receive the dictionary of companies list an return the name of the companies in a string company1, company2"""

    if len(list_companies) != 0:

        name_company = ''

        for company_dict in list_companies:

            name_company = name_company + ',' + company_dict['name']

        name_company = name_company[1:]

    else:

        name_company = ''

    return name_company



def get_bool_production_companies(list_companies):

    if list_companies != '':

        return 1

    else:

        return 0
movies_df['production_companies'] = movies_df['production_companies'].progress_apply(get_production_companies)

movies_df['bool_production_company'] = movies_df['production_companies'].progress_apply(get_bool_production_companies)
def get_production_countries(list_countries):

    """receive the dictionary of countries an return then name of the countries in a string"""

    if len(list_countries) != 0:

        name_country = ''

        for country_dict in list_countries:

            name_country = name_country + ',' + country_dict['name']

        name_country = name_country[1:]

    else:

        name_country = ''

        

    return name_country



def get_bool_production_countries(list_countries):

    if list_countries != '':

        return 1

    else:

        return 0
movies_df['production_countries'] = movies_df['production_countries'].progress_apply(get_production_countries)

movies_df['bool_production_countries'] = movies_df['production_countries'].progress_apply(get_bool_production_countries)
def get_spoken_languages(list_languages):

    if len(list_languages) != 0:

        name_language = ''

        for language_dict in list_languages:

            name_language = name_language + ',' + language_dict['iso_639_1']

        name_language = name_language[1:]

    else:

        name_language = ''

    return name_language

    

def get_bool_spoken_languages(list_languages):

    if list_languages != '':

        return 1

    else:

        return 0
movies_df['spoken_languages'] = movies_df['spoken_languages'].progress_apply(get_spoken_languages)

movies_df['bool_spoken_languages'] = movies_df['spoken_languages'].progress_apply(get_bool_spoken_languages)
movies_df.columns.values
movies_df.index = range(movies_df.shape[0])
movies_df.head(7)
print(movies_df['budget'].describe())

sns.boxplot(movies_df['budget'])

plt.show()
print(movies_df['budget'].value_counts())
def replace_zero_budget(budget):

    if budget == 0:

        return 1

    else:

        return budget

            
movies_df['budget'] = movies_df['budget'].progress_apply(replace_zero_budget)
print(movies_df['popularity'].describe())

sns.boxplot(movies_df['popularity'])

plt.show()
movies_df['popularity'].sort_values()
print(movies_df['revenue'].describe())

sns.boxplot(movies_df['revenue'])

plt.show()
movies_df['revenue'].value_counts()
def replace_zero_revenue(revenue):

    if revenue == 0:

        return 1

    else:

        return revenue
movies_df['revenue'] = movies_df['revenue'].progress_apply(replace_zero_revenue)
movies_df['runtime'].isnull().sum()
movies_df['runtime'].fillna(value = 81, inplace=True)
print(movies_df['runtime'].describe())

sns.boxplot(movies_df['runtime'])

plt.show()
def eliminate_outliers_runtime(runtime):

    if runtime//60 > 4:

        return 3*60

    else:

        return runtime
movies_df['runtime'] = movies_df['runtime'].progress_apply(eliminate_outliers_runtime)
print(movies_df['vote_average'].describe())

sns.boxplot(movies_df['vote_average'])

plt.show()
print(movies_df['vote_count'].describe())

sns.boxplot(movies_df['vote_count'])

plt.show()
import nltk

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from collections import Counter

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk.stem import PorterStemmer 

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics.pairwise import cosine_similarity

import re 



ps = PorterStemmer()

lem = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))
nltk_movies_df = movies_df.select_dtypes(include = ['object']).copy()
nltk_movies_df = nltk_movies_df.drop(columns = ['homepage', 'imdb_id', 'poster_path'])
def pre_process(text):

    text = text.lower() #lowerCase

    text = re.sub("&lt;/?.*?&gt;", "&lt;&gt; ", text) #remove tags

#     text = re.sub("(\\d|\\W)+", " ", text) #remove special characters and digits

    return text

    
for col in nltk_movies_df.columns:

    nltk_movies_df[f'{col}'] = nltk_movies_df[f'{col}'].progress_apply(pre_process)
for i in nltk_movies_df.columns:

    print(i, end = ' ')
nltk_movies_df['combine'] = ''

nltk_movies_df['combine'] = ' '+ nltk_movies_df['genres']+ ' '+ nltk_movies_df['original_language']+ ' '+ nltk_movies_df['original_title']+ ' '\

+ nltk_movies_df['overview']+ ' '+ nltk_movies_df['production_companies']+ ' '+ nltk_movies_df['production_countries']+ ' '+ nltk_movies_df['release_date']+ ' '\

+ nltk_movies_df['spoken_languages']+ ' '+ nltk_movies_df['status']+ ' '+ nltk_movies_df['tagline']
def get_tokens(combine):

    tokens = word_tokenize(combine)

    filtered = [w for w in tokens if not w in stop_words]

    tokenize_sentence = ''

    for f in filtered:

        tokenize_sentence += ' '+ ps.stem(f)

    return tokenize_sentence
nltk_movies_df['combine'].values[0]
get_lemmatize(nltk_movies_df['combine'].values[0])
nltk_movies_df['tokens_combine'] = nltk_movies_df['combine'].progress_apply(get_tokens)
nltk_movies_df['combine']
nltk_movies_df['tokens_combine']
movies_df.to_csv('all_movie_data.csv')

nltk_movies_df.to_csv('nltk_movies.csv')
# def sort_coo(coo_matrix):

#     tuples = zip(coo_matrix.col, coo_matrix.data)

#     return sorted(tuples, key=lambda x: (x[1], x[0]), reverse = True)



# def extract_topn_from_vector(feature_names, sorted_items, topn = 10):

#     sorted_items = sorted_items[:topn]

    

#     score_vals = []

#     feature_vals = []

    

#     for idx, score in sorted_items:

        

#         score_vals.append(round(score, 3))

#         feature_vals.append(feature_names[idx])

    

#     results = {}

#     for idx in range(len(feature_vals)):

#         results[feature_vals[idx]] = score_vals[idx]

#     return results
# indexes = movies_df[movies_df['title'].str.contains('Batman')].index.values
# nltk_movies_df['keywords'] = ''

# with tqdm(total = len(indexes)) as pbar:

#     for i in indexes:

#         cv = CountVectorizer()

#         word_count_vector = cv.fit_transform([nltk_movies_df['tokens_combine'][i]])

#         tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

#         tfidf_transformer.fit(word_count_vector)

#         feature_names = cv.get_feature_names()

#         doc = nltk_movies_df['tokens_combine'][i]

#         tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

#         sorted_items = sort_coo(tf_idf_vector.tocoo())

#         keywords = extract_topn_from_vector(feature_names,sorted_items, 30)

#         for k in keywords:

#             nltk_movies_df['keywords'][i] += ' '+ k

#         pbar.update(1) 
# nltk_movies_df['keywords'] = ''

# with tqdm(total = nltk_movies_df.shape[0]) as pbar:

#     for i in range(nltk_movies_df.shape[0]):

#         cv = CountVectorizer()

#         word_count_vector = cv.fit_transform([nltk_movies_df['tokens_combine'][i]])

#         tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

#         tfidf_transformer.fit(word_count_vector)

#         feature_names = cv.get_feature_names()

#         doc = nltk_movies_df['tokens_combine'][i]

#         tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

#         sorted_items = sort_coo(tf_idf_vector.tocoo())

#         keywords = extract_topn_from_vector(feature_names,sorted_items, 30)

#         for k in keywords:

#             nltk_movies_df['keywords'][i] += ' '+ k

#         pbar.update(1) 
# proof = nltk_movies_df[nltk_movies_df['keywords'] != ''].copy()

# proof.index = range(proof.shape[0])
# nltk_movies_df[nltk_movies_df['keywords'] != '']['keywords'][0]
# nltk_movies_df['tokens_combine'][0]
# cv = CountVectorizer()

# word_count_vector = cv.fit_transform(proof['tokens_combine'].tolist())
# nltk_movies_df.columns
# cv.get_feature_names()
# import sklearn.preprocessing as pp

# def faster_cosine_similarities(mat):

#     col_normed_mat = pp.normalize(mat.tocsc(), axis=0)

#     return col_normed_mat.T * col_normed_mat
# cosine_similarity = faster_cosine_similarities(word_count_vector)
# proof[proof['title'].str.contains('batman')].head(30)
# movies_df[movies_df['title'].str.contains('Batman')].head(30)
# similar_movies = list(enumerate(cosine_similarity[1].toarray()[0]))

# sort_similar_movie = sorted(similar_movies, key = lambda x: x[1], reverse = True)
# sort_similar_movie
# def get_tittle_from_index(index, df):

#     return df[df.index == index]['title'].values[0]



# def get_index_from_title(title, df):

#     return df[df['title'] == title].index.values[0]



# def get_id_from_index(index, df):

#     return df[df.index == index]['id'].values[0]



# def get_poster_path_from_index(index, df):

#     return df[df.index == index]['poster_path'].values[0]
# for similar in sort_similar_movie:

#     if similar[1] > 0:

#         print(f'{get_tittle_from_index(similar[0], proof)} ### similarity of {round(similar[1], 3)}')

#         ids = get_id_from_index(similar[0], proof)

#         print(get_poster_path_from_index(similar[0], proof))

#         display(Image(url= f"https://image.tmdb.org/t/p/w500///{get_poster_path_from_index(similar[0], proof)}", width=100, height=100))

#     else:

#         break
# from IPython.display import Image, display

# from IPython.core.display import HTML

# display(Image(url= f"https://image.tmdb.org/t/p/w500///v4ftSPtKJO2CkzlHOopfWp5i7MU.jpg", width=100, height=100))

    


# movies_df.to_csv('all_movie_data.csv')
# movie_extra[movie_extra['title'] == 'Batman']