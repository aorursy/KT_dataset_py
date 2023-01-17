import pandas as pd

import numpy as np 

import numpy as np

from textblob import TextBlob

import nltk

import string

import random

import tensorflow as tf

import os

import io

import sys

import os

import numpy as np

import pandas as pd

import nltk

import gensim

import csv, collections

from textblob import TextBlob

from sklearn.utils import shuffle

from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes

from nltk.corpus import stopwords
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

subm = pd.read_csv('../input/sample_submission.csv')

test.head()
print("Number of sentences in train data is {}".format(train.shape[0]))

print("Number of NAs in train_data {}".format(train.isnull().sum()))

categories = ['toxic','severe_toxic','obscene','threat','insult', 'identity_hate']

sanity = pd.DataFrame(train.groupby(categories)['id'].count())

sanity_copy = sanity.copy()

sanity.reset_index(inplace = True)

if sanity.shape[0] == 5:

    print("One sentence falls into one category")

else:

    print("They want us to train multiple models NN or GB, OR Gaussian miture models")   
train_df = train.copy() # just saving copy of train data 

test_df = test.copy()

eng_stopwords = set(stopwords.words("english"))

import time

start = time.time()

def remove_noise(row):

    """function to remove unnecessary noise from the data - sentences"""

    try:

        text = row['comment_text']

        text_splited = text.split(' ')

        text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

        noise_words = ['\n', '\n\n']

        text_splited = [''.join(c for c in s if c not in noise_words) for s in text_splited]

        text_splited = [s for s in text_splited if s]

        return(text_splited)

    except:

        return(row['comment_text'])

    

    

    

def grams_features(train_df, test_df):

    """function to extract grams features for a given sentence"""

    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

    full_tfidf = tfidf_vec.fit_transform(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())

    train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())

    test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())

    return(train_tfidf, test_tfidf)

        

end = time.time()    

print("Time taken in tf-idf is {}.".format(end-start))
train_df['processed_text'] = train_df.apply(lambda row: remove_noise(row), axis = 1)

test_df['processed_text'] = test_df.apply(lambda row: remove_noise(row), axis = 1)

train_df.head()
train_df.dropna(inplace = True)

test_df.dropna(inplace = True)

start = time.time()

train_tfidf, test_tfidf = grams_features(train_df, test_df)

end = time.time()

print("Time taken in tf-idf is {}.".format(end-start))
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)



x=train_tfidf.sign()

test_x = test_tfidf.sign()

from sklearn.linear_model import LogisticRegression

def get_mdl(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=0.1, dual=True)

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r



preds = np.zeros((len(test_df), len(label_cols)))



for i, j in enumerate(label_cols):

    print('fit', j)

    m,r = get_mdl(train_df[j])

    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)