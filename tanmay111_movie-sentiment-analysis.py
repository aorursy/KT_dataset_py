import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
pd.set_option('display.max_columns',None)
movie = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
movie_test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
movie_test.head()
movie.head()
movie.shape
movie.nunique()
movie.isnull().sum()
from nltk.stem.porter import *

stemmer = PorterStemmer()
def text_cleaning(df):

    df = df.str.replace("[^a-zA-Z#]", " ")

    return df
movie['Phrase'] = text_cleaning(movie['Phrase'])
movie_test['Phrase'] = text_cleaning(movie_test['Phrase'])
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
x = movie['Phrase']

y = movie['Sentiment']
xtest = movie_test['Phrase']
# Logistic

phrase_lr_cv = Pipeline([('cv',CountVectorizer()),

                         ('lr', LogisticRegression()),

])



# Naïve Bayes:

phrase_nb_cv = Pipeline([('cv',CountVectorizer()),

                         ('nb', MultinomialNB()),

])



# Linear SVC:

phrase_svc_cv = Pipeline([('cv',CountVectorizer()),

                         ('review', LinearSVC()),

])
# Logistic

phrase_lr_tf = Pipeline([('tfidf', TfidfVectorizer()),

                         ('review', LogisticRegression()),

])



# Naïve Bayes:

phrase_nb_tf = Pipeline([('tfidf', TfidfVectorizer()),

                         ('review', MultinomialNB()),

])



# Linear SVC:

phrase_svc_tf = Pipeline([('tfidf', TfidfVectorizer()),

                         ('review', LinearSVC()),

])
def model(obj):

    ypred = obj.fit(x,y).predict(xtest)

    data = pd.DataFrame({'PhraseId':movie_test.PhraseId,'Sentiment':ypred})

    return(data)

    
output = model(phrase_svc_tf)
output.to_csv('test.csv', index=False)