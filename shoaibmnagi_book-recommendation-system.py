import numpy as np

import pandas as pd

import nltk

import json

import re

import csv

from tqdm import tqdm

pd.set_option('display.max_colwidth', 300)



data = []



with open("/kaggle/input/cmu-book-summary-dataset/booksummaries.txt", 'r') as f:

    reader = csv.reader(f, dialect='excel-tab')

    for row in tqdm(reader):

        data.append(row)
book_index = []

book_id = []

book_author = []

book_name = []

summary = []

genre = []

a = 1

for i in tqdm(data):

    book_index.append(a)

    a = a+1

    book_id.append(i[0])

    book_name.append(i[2])

    book_author.append(i[3])

    genre.append(i[5])

    summary.append(i[6])



df = pd.DataFrame({'Index': book_index, 'ID': book_id, 'BookTitle': book_name, 'Author': book_author,

                       'Genre': genre, 'Summary': summary})

df.head()
df.isna().sum()



df = df.drop(df[df['Genre'] == ''].index)

df = df.drop(df[df['Summary'] == ''].index)





genres_cleaned = []

for i in df['Genre']:

    genres_cleaned.append(list(json.loads(i).values()))

df['Genres'] = genres_cleaned



def clean_summary(text):

    text = re.sub("\'", "", text)

    text = re.sub("[^a-zA-Z]"," ",text)

    text = ' '.join(text.split())

    text = text.lower()

    return text



df['clean_summary'] = df['Summary'].apply(lambda x: clean_summary(x))

df.head(2)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords





df['GenreString'] = df['Genres'].apply(lambda x: ' '.join(x))



#get a combined text that includes author's name and associated genres

df["combined_text"] = df["clean_summary"] + " " + df["Author"] + " " + df["GenreString"]





"""stopwords = stopwords.words('english')

df['text_without_stopwords'] = df['combined_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

cv = CountVectorizer()

count_matrix = cv.fit_transform(df['text_without_stopwords'])"""



tf = TfidfVectorizer(analyzer = "word", ngram_range=(1,2), min_df=0, stop_words='english')



tfidf_matrix = tf.fit_transform(df['combined_text'])



cosine =  cosine_similarity(tfidf_matrix, tfidf_matrix)







def get_title_from_index(Index):

    return df[df.Index == Index]["BookTitle"].values[0]

def get_index_from_title(BookTitle):

    return df[df.BookTitle == BookTitle]["Index"].values[0]



def get_recommendations(book):

    book_index = get_index_from_title(book)

    similar_books = list(enumerate(cosine[book_index]))

    sortedbooks = sorted(similar_books, key = lambda x:x[1], reverse=True)[1:]

    i = 0

    for book in sortedbooks:

        print(get_title_from_index(book[0]) + " by " + df.Author[df["Index"] == book[0]])

        i = i+1

        if i>10:

            break
print(get_recommendations("The Stand"))
print(get_recommendations("A Clockwork Orange"))
print(get_recommendations("Dune"))
print(get_recommendations("Oliver Twist"))
print(get_recommendations('White Noise'))