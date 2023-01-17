import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import requests

import re

import string

import random



import warnings

warnings.filterwarnings("ignore")



from nltk.corpus import stopwords

from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import RegexpTokenizer

from PIL import Image

from io import BytesIO

from sklearn.metrics.pairwise import cosine_similarity



pd.set_option('display.max_rows', 500)
raw_books = pd.read_csv('../input/7k-books-with-metadata/books.csv')

raw_books.head(3)
print('ROWS: ', raw_books.shape[0])

print('COLUMNS: ', raw_books.shape[1])
raw_books['categories'].nunique()
raw_books['categories'].unique()
raw_books['categories'].value_counts().head(10)
books = raw_books.loc[raw_books['categories'].isin(['Fiction', 'Juvenile Fiction', 

                                                    'Biography & Autobiography', 'History'])]
books.tail(5)
plt.figure(figsize=(10, 6))

sns.set_style('darkgrid')

sns.countplot(x=books['categories'], palette='Blues_r', edgecolor='black')



plt.show()
print(books['title'] [120])

print(books['description'] [120])

print('\n\n')

print(books['title'] [200])

print(books['description'] [200])


books.dropna(subset=['description'], inplace=True)
books['description'].isna().sum()
# USING REGULAR EXPRESSIONS (REGEX)

books = books[~books.description.str.contains('[0-9].*[0-9].*[printing]')]
# CONVERT DESCRIPTION INTO VECTORS AND USE BIGRA,

tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase=False)

tfidf_matrix = tf.fit_transform(books['description'])

total_words = tfidf_matrix.sum(axis=0)



# WORK FREQUENCY

freq = [(word, total_words[0, index]) for word, index in tf.vocabulary_.items()]

freq = sorted(freq, key=lambda x: x[1], reverse=True)



# CREATE A PANDAS DATAFRA,E

bigram = pd.DataFrame(freq)

bigram.rename(columns = {0:'bigram', 1: 'count'}, inplace = True) 



# TOP 20 BIGRAMS

bigram = bigram.head(20)



# PLOT BARPLOT

plt.figure(figsize=(8, 8))

sns.barplot(x=bigram['count'], y=bigram['bigram'], color='blue')

plt.show()
# REMOVE NON ASCII CHARACTERS

def remove_non_ascii(string):

    return "".join(c for c in string if ord(c) < 128)



# MAKE DESCRIPTION TEXT LOWER CASE

def make_lower_case(text):

    return text.lower()



# REMOVE STOP WORDS

def remove_stop_words(text):

    text = text.split()

    stops = set(stopwords.words('english'))

    text = [word for word in text if not word in stops]

    text = " ".join(text)

    return text



# REMOVE PUNCTUATIONS

def remove_punctuation(text):

    tokenizer = RegexpTokenizer(r'\w+')

    text = tokenizer.tokenize(text)

    text = " ".join(text)

    return text



# REMOVE HTML CODES

def remove_html(text):

    html_pattern = re.compile('<.*?>')

    return html_pattern.sub(r'', text)
books['cleaned_description'] = books['description'].apply(remove_non_ascii)

books['cleaned_description'] = books.cleaned_description.apply(make_lower_case)

books['cleaned_description'] = books.cleaned_description.apply(remove_stop_words)

books['cleaned_description'] = books.cleaned_description.apply(remove_punctuation)

books['cleaned_description'] = books.cleaned_description.apply(remove_html)
def recommend(title, category):

    

    # MATCH THE CATEGORY WITH THE COLUMN "CATEGORIES" OF THE DATASET

    data = books.loc[books['categories'] == category] 

    # RESET INDEX

    data.reset_index(level = 0, inplace = True) 

    

    # INDEX TO A PANDAS SERIES

    indices = pd.Series(data.index, index = data['title'])

    

    # CONVERT THE BOOK TITLE INTO VECTORS AND USE BIGRAM

    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')

    tfidf_matrix = tf.fit_transform(data['title'])

    

    # CALCULATE THE SIMILARITY MEASURE

    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    

    # GET THE INDEX OF ORIGINAL TITLE

    index = indices[title]

    

    # PAIRWISE SIMILARITY SCORES

    similarity = list(enumerate(similarity[index]))

    # SORT THE BOOKS

    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)

    

    # GET TOP 5 MOST SIMILAR BOOKS

    similarity  = similarity [1:6]

    

    # INDICES OF TOP 5

    book_indices = [i[0] for i in similarity]



    # TOP 5 RECOMMENDATION

    rec = data[['title', 'thumbnail']].iloc[book_indices]

    

    # PRINT THE BOOKS TITLE

    print(rec['title'])

    

    # PRINT THE TOP 5 BOOK COVER

    for i in rec['thumbnail']:

        response = requests.get(i)

        img = Image.open(BytesIO(response.content))

        plt.figure()

        plt.imshow(img)
# TEST

recommend("A People's History of the United States", "History")
def recommend(title, category):

    

    # MATCH THE CATEGORY WITH THE COLUMN "CATEGORIES" OF THE DATASET

    data = books.loc[books['categories'] == category] 

    # RESET INDEX

    data.reset_index(level = 0, inplace = True) 

    

    # INDEX TO A PANDAS SERIES

    indices = pd.Series(data.index, index = data['title'])

    

    # CONVERT THE BOOK TITLE INTO VECTORS AND USE BIGRAM

    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')

    tfidf_matrix = tf.fit_transform(data['cleaned_description'])

    

    # CALCULATE THE SIMILARITY MEASURE

    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    

    # GET THE INDEX OF ORIGINAL TITLE

    index = indices[title]

    

    # PAIRWISE SIMILARITY SCORES

    similarity = list(enumerate(similarity[index]))

    # SORT THE BOOKS

    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)

    

    # GET TOP 5 MOST SIMILAR BOOKS

    similarity  = similarity [1:6]

    

    # INDICES OF TOP 5

    book_indices = [i[0] for i in similarity]



    # TOP 5 RECOMMENDATION

    rec = data[['title', 'thumbnail']].iloc[book_indices]

    

    # PRINT THE BOOKS TITLE

    print(rec['title'])

    

    # PRINT THE TOP 5 BOOK COVER

    for i in rec['thumbnail']:

        response = requests.get(i)

        img = Image.open(BytesIO(response.content))

        plt.figure()

        plt.imshow(img)
# TEST

recommend("Taken at the Flood", "Fiction")