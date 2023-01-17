# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import nltk

import regex as re

from sklearn import linear_model

import pdb

from nltk.stem.porter import *

import category_encoders as ce

import xgboost as xgb



train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

output = train_df['target']

input_df = train_df.drop( ['id','target'], axis = 1)

input_df['keyword'].fillna("UNAVAILABLE", inplace = True)

input_df['location'].fillna("UNKNOWN", inplace = True)

# remove user-names

input_df['text'] = input_df['text'].str.replace("@[\w]*","")

# remove special characters, numbers, punctuations

input_df['text'] = input_df['text'].str.replace("[^a-zA-Z#]", " ")

input_df.head()
from nltk.corpus import stopwords



stemmer = PorterStemmer()

tokenized_tweet = input_df['text'].apply(lambda x: x.split())

stop_words = set(stopwords.words('english'))

stemmed = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x if i not in stop_words])



tweet = stemmed.apply(lambda x: ' '.join(x))



tweet
# convert tweet to tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf_tweet = tfidf_vectorizer.fit_transform(tweet)

tfidf_tweet
#woe encoder



col_names = ['keyword', 'location']

woe_encoder = ce.WOEEncoder(cols= col_names)

woe_encoded_train = woe_encoder.fit_transform(input_df[col_names], output).add_suffix("woe")

xtrain_tweet, xvalid_tweet, ytrain_tweet, yvalid_tweet = train_test_split(tfidf_tweet, output, random_state=42, test_size=0.3)

clf = linear_model.RidgeClassifier()



trained_tweet_model = clf.fit(xtrain_tweet,ytrain_tweet)

trained_tweet_proba = trained_tweet_model.decision_function(xtrain_tweet)





xtrain_lockey, xvalid_lockey, ytrain_lockey, yvalid_lockey = train_test_split(woe_encoded_train, output, random_state=42, test_size=0.3)



clf = xgb.XGBRegressor(objective="binary:logistic", random_state=42)

trained_lockey_model = clf.fit(xtrain_lockey, ytrain_lockey)

trained_lockey_proba = trained_lockey_model.predict(xtrain_lockey)



#Merging

next_input = zip(trained_tweet_proba, trained_lockey_proba)

x = [data for data in list(next_input)]

clf = linear_model.RidgeClassifier()

trained_final_model = clf.fit(x, ytrain_tweet)

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test['keyword'].fillna("UNAVAILABLE", inplace=True)

test['location'].fillna("UNKNOWN", inplace=True)

# remove user-names

test['text'] = test['text'].str.replace("@[\w]*", "")

# remove special characters, numbers, punctuations

test['text'] = test['text'].str.replace("[^a-zA-Z#]", " ")

stemmer = PorterStemmer()

tokenized_tweet = test['text'].apply(lambda x: x.split())

stemmed = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])



tweet = stemmed.apply(lambda x: ' '.join(x))

# convert tweet to tfidf

tfidf_tweet = tfidf_vectorizer.transform(tweet)





col_names = ['keyword', 'location']

test_lockey = woe_encoder.transform(test[col_names]).add_suffix("woe")



test_tweet_proba = trained_tweet_model.decision_function(tfidf_tweet)



test_lockey_proba = trained_lockey_model.predict(test_lockey)



# Merging

next_input = zip(test_tweet_proba, test_lockey_proba)

x = [data for data in next_input]

out = trained_final_model.predict(x)



test['target'] = out

cols = ['id','target']

test[cols].to_csv("submit.csv", index = False)

test[cols].head()