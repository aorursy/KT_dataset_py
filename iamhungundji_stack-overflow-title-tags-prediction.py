import pandas as pd

import numpy as np



import nltk

import re

import csv



import matplotlib.pyplot as plt

%matplotlib inline



from tqdm import tqdm



import bq_helper

from bq_helper import BigQueryHelper



import warnings

warnings.filterwarnings('ignore', message=r'Label not .* is present in all training examples.')



pd.set_option('display.max_colwidth', 300)
from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import train_test_split



from sklearn.preprocessing import MultiLabelBinarizer



from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier



from sklearn.metrics import f1_score, accuracy_score
stack_overflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                         dataset_name="stackoverflow")
bg = BigQueryHelper("bigquery-public-data", "stackoverflow")

bg.list_tables()
bg.head("stackoverflow_posts", num_rows=1)
bg.table_schema("stackoverflow_posts")
query = """

        SELECT 

            id, title , tags 

        FROM 

            `bigquery-public-data.stackoverflow.stackoverflow_posts`

        WHERE

            title NOT LIKE '%None%' AND 

            (tags LIKE '%|python|%' OR tags LIKE '%|c#|%' OR

            tags LIKE '%|java|%' OR tags LIKE '%|r|%' OR

            tags LIKE '%|android|%' OR tags LIKE '%|html|%' OR

            tags LIKE '%|c++|%' OR tags LIKE '%|sql|%' OR

            tags LIKE '%|c|%' OR tags LIKE '%kotlin%') AND 

            LENGTH(tags) < 20

        LIMIT

             10000;

        """



data = stack_overflow.query_to_pandas(query)



data_copy = data.copy()



data.head()
data.title = data.title.str.replace('<[^<]+?>','')

data.title = data.title.str.replace('http','')

data.title = data.title.str.replace('[^\w\s]','')

data.title = data.title.str.lower()

data.head()
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



def remove_stopwords(text):

    no_stopword_text = [w for w in text.split() if not w in stop_words]

    return ' '.join(no_stopword_text)



data['title'] = data['title'].apply(lambda x: remove_stopwords(x))

data.head()
data['tags'] = data['tags'].str.split('|')

data.head()
multilabel_binarizer = MultiLabelBinarizer()

multilabel_binarizer.fit(data['tags'])



y = multilabel_binarizer.transform(data['tags'])
x_train, x_val, ytrain, yval = train_test_split(data['title'],

                                                y, test_size=0.2)
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, max_features=5000)

xtrain = tfidf_vectorizer.fit_transform(x_train)

xval = tfidf_vectorizer.transform(x_val)
lr = LogisticRegression()

classifier = OneVsRestClassifier(lr)

classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xval)

f1_score(yval, y_pred, average="micro"), accuracy_score(yval, y_pred)
pred_prob = classifier.predict_proba(xval)

t = 0.3

predp = (pred_prob >= t).astype(int)

f1_score(yval, predp, average="micro"), accuracy_score(yval, predp)
def predict(m):

    m = remove_stopwords(m)

    m_vec = tfidf_vectorizer.transform([m])

    pred_prob = classifier.predict_proba(m_vec)

    t = 0.3

    predp = (pred_prob >= t).astype(int)

    #m_pred = classifier.predict(m_vec)

    return multilabel_binarizer.inverse_transform(predp)
for i in range(10):

    k = x_val.sample(1).index[0]

    print("Title: ", data_copy['title'][k],

          "\nPredicted tags: ", predict(x_val[k])),

    print("Actual tags: ",data['tags'][k], "\n")