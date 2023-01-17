# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import re

import nltk

from nltk.corpus import stopwords

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.stem.porter import *

from string import punctuation

import copy

import itertools



import sklearn.ensemble

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

head_count = 2

stops = set(stopwords.words("english"))

accuracy_lst = list()

logloss_lst = list()
auth_vc = train_data['author'].value_counts()

train_data['author_lbl'] = train_data['author'].map({

    'EAP': 1,

    'MWS': 2,

    'HPL': 3

})
def tokenize(text):

    word_tokens = nltk.word_tokenize(text)

    word_tokens = [w for w in word_tokens if not w in stops]

    word_tokens = list(map(str.lower, word_tokens))

    return word_tokens



def get_bag_of_words(row):

    return Counter(row['tokens'])



def total_punctuations(row):

    return len([p for p in row['tokens'] if p in list(punctuation)])

                   

def total_stopwords(row):

    return len([p for p in row['tokens'] if p in stops])



def get_random_idx(df):

    return np.random.choice(df.index.values)
def generate_text_features(df):

    df['n_words']               = df.apply(lambda row: len(row['text']), axis=1)

    df['tokens']                = df.apply(lambda row: tokenize(row['text']), axis=1)

    df['n_tokens']              = df['tokens'].map(len)

    df['bow']                   = df.apply(lambda row: get_bag_of_words(row), axis=1)

    df['n_puncts']              = df.apply(lambda row: total_punctuations(row), axis=1)

    df['#,']                    = df.apply(lambda row: row['bow'][','], axis=1)

    df['#;']                    = df.apply(lambda row: row['bow'][';'], axis=1)

    return df
train_data = generate_text_features(train_data)
punctuation
features = ['n_puncts', '#;']

X = train_data[features]



y = train_data['author_lbl']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(random_state=0)

clf = clf.fit(X_train, y_train)



y_proba = clf.predict_proba(X_test)

log_loss_score = log_loss(y_test, y_proba)

logloss_lst.append("{:>5.3f}".format(log_loss_score))



y_predict = clf.predict(X_test)

predict_df = pd.DataFrame({'actual': y_test.ravel(), 'predicts': y_predict})

predict_df['score'] = predict_df.apply(lambda x: 1 if x['actual'] == x['predicts'] else 0, axis=1)

accuracy = predict_df['score'].sum() / len(predict_df)

accuracy = accuracy * 100

accuracy_lst.append("{:>5.3f}".format(accuracy))



print("Logloss ", ", ".join(logloss_lst))

print("Accuracy ", ", ".join(accuracy_lst))