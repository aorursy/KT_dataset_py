import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import json

import nltk

import re

import csv

import matplotlib.pyplot as plt

from tqdm import tqdm

%matplotlib inline

pd.set_option('display.max_colwidth', 300)
from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import train_test_split



from sklearn.preprocessing import MultiLabelBinarizer



from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier



from sklearn.metrics import f1_score, accuracy_score
data = []



with open("/kaggle/input/cmu-book-summary-dataset/booksummaries.txt", 'r') as f:

    reader = csv.reader(f, dialect='excel-tab')

    for row in tqdm(reader):

        data.append(row)
book_id = []

book_name = []

summary = []

genre = []



for i in tqdm(data):

    book_id.append(i[0])

    book_name.append(i[2])

    genre.append(i[5])

    summary.append(i[6])



books = pd.DataFrame({'book_id': book_id, 'book_name': book_name,

                       'genre': genre, 'summary': summary})

books.head(2)
books.shape
books.drop(books[books['genre']==''].index, inplace=True)

books[books['genre']=='']
json.loads(books['genre'][0]).values()
genres = []

for i in books['genre']:

    genres.append(list(json.loads(i).values()))

books['genre_new'] = genres
all_genres = sum(genres,[])

len(set(all_genres))
def clean_summary(text):

    text = re.sub("\'", "", text)

    text = re.sub("[^a-zA-Z]"," ",text)

    text = ' '.join(text.split())

    text = text.lower()

    return text
books['clean_summary'] = books['summary'].apply(lambda x: clean_summary(x))

books.head(2)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):

    no_stopword_text = [w for w in text.split() if not w in stop_words]

    return ' '.join(no_stopword_text)



books['clean_summary'] = books['clean_summary'].apply(lambda x: remove_stopwords(x))
multilabel_binarizer = MultiLabelBinarizer()

multilabel_binarizer.fit(books['genre_new'])



y = multilabel_binarizer.transform(books['genre_new'])
x_train, x_val, ytrain, yval = train_test_split(books['clean_summary'],

                                              y, test_size=0.2)
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

xtrain = tfidf_vectorizer.fit_transform(x_train)

xval = tfidf_vectorizer.transform(x_val)
lr = LogisticRegression()

clf = OneVsRestClassifier(lr)

clf.fit(xtrain, ytrain)
y_pred = clf.predict(xval)

f1_score(yval, y_pred, average="micro"), accuracy_score(yval, y_pred)
pred_prob = clf.predict_proba(xval)
t = 0.3

predp = (pred_prob >= t).astype(int)

f1_score(yval, predp, average="micro"), accuracy_score(yval, predp)
def predict(m):

    m = clean_summary(m)

    m = remove_stopwords(m)

    m_vec = tfidf_vectorizer.transform([m])

    m_pred = clf.predict(m_vec)

    return multilabel_binarizer.inverse_transform(m_pred)
for i in range(10):

    k = x_val.sample(1).index[0]

    print("Book: ", books['book_name'][k], 

          "\nPredicted genre: ", predict(x_val[k])) ,

    print("Actual genre: ",books['genre_new'][k], "\n")