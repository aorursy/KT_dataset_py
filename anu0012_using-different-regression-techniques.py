# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor

from sklearn.linear_model import LogisticRegression,SGDClassifier, LinearRegression

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import PorterStemmer

import nltk

from nltk import word_tokenize, ngrams

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import xgboost as xgb

np.random.seed(25)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_table("../input/train.tsv")

test = pd.read_table("../input/test.tsv")
train.head()
def get_cat_1(x):

    txt = str(x)

    y = txt.split('/')

    return y[0]
def get_cat_2(x):

    txt = str(x)

    y = txt.split('/')

    if len(y) == 1:

        return 'None'

    return y[1]
def get_cat_3(x):

    txt = str(x)

    y = txt.split('/')

    if len(y) == 1:

        return 'None'

    return y[2]
train['category_1'] = train['category_name'].map(lambda x: get_cat_1(x))

train['category_2'] = train['category_name'].map(lambda x: get_cat_2(x))

train['category_3'] = train['category_name'].map(lambda x: get_cat_3(x))
test['category_1'] = test['category_name'].map(lambda x: get_cat_1(x))

test['category_2'] = test['category_name'].map(lambda x: get_cat_2(x))

test['category_3'] = test['category_name'].map(lambda x: get_cat_3(x))
# function to clean data

import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from string import punctuation



stops = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',

              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',

              'Is','If','While','This']

# punct = list(string.punctuation)

# punct.append("''")

# punct.append(":")

# punct.append("...")

# punct.append("@")

# punct.append('""')

def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):

    

    txt = str(text)

    

    txt = re.sub(r'[^A-Za-z\s]',r' ',txt)

    

     

    if lowercase:

        txt = " ".join([w.lower() for w in txt.split()])

        

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in stops])

    if stemming:

        st = PorterStemmer()

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])



    return txt
# clean text

train['item_description'] = train['item_description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))

test['item_description'] = test['item_description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))



train['name'] = train['name'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))

test['name'] = test['name'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))
tfidfvec1 = CountVectorizer(analyzer='word', ngram_range = (1,1),stop_words='english',max_features=10)

tfidfdata1 = tfidfvec1.fit_transform(train['item_description'])
tfidfvec2 = CountVectorizer(analyzer='word', ngram_range = (1,1),stop_words=None,max_features=10)

tfidfdata2 = tfidfvec2.fit_transform(train['name'])
# create dataframe for features

tfidf_df = pd.concat([pd.DataFrame(tfidfdata1.todense()), (pd.DataFrame(tfidfdata2.todense()))], axis=1)
tfidf_df.columns = ['col' + str(x) for x in range(20)]   # total features
train = pd.concat([train,tfidf_df], axis=1)
tfidfvec4 = CountVectorizer(analyzer='word', ngram_range = (1,1),stop_words='english',max_features=10)

tfidfdata4 = tfidfvec4.fit_transform(test['item_description'])
tfidfvec5 = CountVectorizer(analyzer='word', ngram_range = (1,1),stop_words=None,max_features=10)

tfidfdata5 = tfidfvec5.fit_transform(test['name'])
# create dataframe for features

tfidf_df2 = pd.concat([pd.DataFrame(tfidfdata4.todense()), (pd.DataFrame(tfidfdata5.todense()))], axis=1)
tfidf_df2.columns = ['col' + str(x) for x in range(20)]   # total test features
test = pd.concat([test,tfidf_df2], axis=1)
train = train.fillna({"brand_name": "other"})

test = test.fillna({"brand_name": "other"})
# One-hot encoding

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train["brand_name"] = lb_make.fit_transform(train["brand_name"])

test["brand_name"] = lb_make.fit_transform(test["brand_name"])



train["category_1"] = lb_make.fit_transform(train["category_1"])

test["category_1"] = lb_make.fit_transform(test["category_1"])



train["category_2"] = lb_make.fit_transform(train["category_2"])

test["category_2"] = lb_make.fit_transform(test["category_2"])



train["category_3"] = lb_make.fit_transform(train["category_3"])

test["category_3"] = lb_make.fit_transform(test["category_3"])
feature_names = [x for x in train.columns if x not in ['train_id','price','test_id','item_description','name','category_name']]
# train = train.astype(str)

# test = test.astype(str)

train[feature_names].isnull().sum(axis=0)
#feature_names = [x for x in train.columns if x not in ['train_id','price','test_id','item_description','name','category_name']]

target = train['price']
#len(feature_names)
# from catboost import CatBoostRegressor



# model_catboost = CatBoostRegressor(eval_metric='RMSE',learning_rate=0.3,verbose=True, iterations=200,depth=4)



# model_catboost.fit(train[feature_names], target, cat_features=[0,  1,  2])



# pred = model_catboost.predict(test[feature_names])
# model = RandomForestRegressor()

# model.fit(train[feature_names],target)
# xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,

#             'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

#             'eval_metric': 'rmse', 'objective': 'reg:linear'}



# model = xgb.train(xgb_par, train[feature_names],target, 80, early_stopping_rounds=20, maximize=False, verbose_eval=20)



model = xgb.XGBRegressor(

                             learning_rate=0.037, max_depth=5, 

                             min_child_weight=20, n_estimators=180,

                             reg_lambda=0.8,booster = 'gbtree',

                             subsample=0.9, silent=1,

                             nthread = -1)



model.fit(train[feature_names], target)

pred = model.predict(test[feature_names])
sub = pd.DataFrame()

sub['test_id'] = test['test_id']

sub['price'] = pred

sub.to_csv('result.csv', index=False)
sub.head()
sub[sub['price'] < 0]