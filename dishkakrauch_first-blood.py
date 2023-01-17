# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#import packages

import re
import logging
import time
import gc
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import *
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
#load data

X_train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#looking what's going on with out dataset

X_train.info()
#first five rows of our dataset

X_train.head(5)
#raw example

X_train['review'][0]
#avoiding HTML tags

BeautifulSoup(X_train["review"][0]).get_text()
#deleting bad symbols

re.sub(pattern="[^a-zA-Z]", repl=' ', string=BeautifulSoup(X_train["review"][0]).get_text())
#function for data cleaning

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text()         
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))
#good example

review_to_words(X_train['review'][0])
%%time
#preprocessiong

num_reviews = X_train["review"].size
print("Cleaning and parsing the training set movie reviews...\n")
X_train_preprocessed = []
for i in iter(range( 0, num_reviews )):
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d" % ( i+1, num_reviews ))                                                                 
    X_train_preprocessed.append( review_to_words( X_train["review"][i] ))
#preprocessed example

X_train_preprocessed[0]
#default vectorizer

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = stopwords.words("english"),
                             ngram_range=(1,5),
                             max_features = 10000)

X_train_preprocessed_vectorized = vectorizer.fit_transform(X_train_preprocessed)

X_train_preprocessed_vectorized = X_train_preprocessed_vectorized.toarray()
print(X_train_preprocessed_vectorized.shape)
X_train_preprocessed_vectorized[0].shape
#vectorizer's features

vectorizer.get_feature_names()
#sum of vectorized features for the first example

sum(X_train_preprocessed_vectorized[0])
#sorted count of words

vocab = vectorizer.get_feature_names()
dist = np.sum(X_train_preprocessed_vectorized, axis=0)

for count, tag in sorted([(count, tag) for tag, count in zip(vocab, dist)], reverse=True):
    print(count, tag)
#deafult random forest classifier

#RFC = RandomForestClassifier(n_estimators=250, 
#                             criterion='gini', 
#                             max_depth=None,
#                             min_samples_split=2, 
#                             min_samples_leaf=1, 
#                             min_weight_fraction_leaf=0.0, 
#                             max_features='auto', 
#                             max_leaf_nodes=None, 
#                             min_impurity_decrease=0.0, 
#                             min_impurity_split=None, 
#                             bootstrap=True, 
#                             oob_score=False, 
#                             n_jobs=-1, 
#                             random_state=0, 
#                             verbose=0, 
#                             warm_start=False, 
#                             class_weight=None)
#%%time
#%env JOBLIB_TEMP_FOLDER=/tmp
#
##using gpu for grid search cv
#
#params = {
#    'n_estimators': [100, 150, 250],
#    'max_depth': [30, 50, 100],
#    'min_samples_split': [.75, .8, .95],
#    'min_samples_leaf': [1, 2, 4]
#}
#
#cv = GridSearchCV(estimator=RFC, param_grid=params, n_jobs=-1, cv=3, verbose=1)
#cv.fit(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
%%time

#fit RFC default model

#RFC.fit(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
#score RFC default model

#RFC.score(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
#load test data

X_test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3)
#looking first 5 rows of test dataset

X_test.head(5)
#check test dataset shape

X_test.shape
%%time

#preprocessing test dataset

num_reviews = X_test["review"].size
print("Cleaning and parsing the testing set movie reviews...\n")
X_test_preprocessed = []
for i in iter(range( 0, num_reviews )):
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d" % ( i+1, num_reviews ))                                                                 
    X_test_preprocessed.append( review_to_words( X_test["review"][i] ))
#final processing of test dataset

X_test_preprocessed_vectorized = vectorizer.transform(X_test_preprocessed)

X_test_preprocessed_vectorized = X_test_preprocessed_vectorized.toarray()
print(X_test_preprocessed_vectorized.shape)
#predict RFC

#pred = RFC.predict(X=X_test_preprocessed_vectorized)
#output of RFC default model

#output = pd.DataFrame( data={"id":X_test["id"], "sentiment":pred} )
#output.to_csv("Bag_of_Words_model_RFC.csv", index=False, quoting=3 )
ETC = ExtraTreesClassifier(n_estimators=250, 
                             criterion='gini', 
                             max_depth=None,
                             min_samples_split=2, 
                             min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=False, 
                             n_jobs=-1,
                             random_state=0, 
                             verbose=0, 
                             warm_start=False, 
                             class_weight=None)
#%%time
#%env JOBLIB_TEMP_FOLDER=/tmp
#
##using gpu for grid search cv
#
#params = {
#    'n_estimators': [100, 150, 250],
#    'max_depth': [30, 50, 100],
#    'min_samples_split': [.75, .8, .95],
#    'min_samples_leaf': [1, 2, 4]
#}
#
#cv = GridSearchCV(estimator=ETC, param_grid=params, n_jobs=-1, cv=3, verbose=1)
#cv.fit(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
#cv.best_estimator_
#ETC = ExtraTreesClassifier(n_estimators=100, 
#                             criterion='gini', 
#                             max_depth=30,
#                             min_samples_split=2, 
#                             min_samples_leaf=1, 
#                             min_weight_fraction_leaf=0.0, 
#                             max_features='auto', 
#                             max_leaf_nodes=None, 
#                             min_impurity_decrease=0.0, 
#                             min_impurity_split=None, 
#                             bootstrap=True, 
#                             oob_score=False, 
#                             n_jobs=-1, 
#                             random_state=0, 
#                             verbose=0, 
#                             warm_start=False, 
#                             class_weight=None)
##%%time
#
##fit ETC default model
#
#ETC.fit(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
##score ETC default model
#
#ETC.score(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
##predict ETC
#
#pred = ETC.predict(X=X_test_preprocessed_vectorized)
##output of ETC default model
#
#output = pd.DataFrame( data={"id":X_test["id"], "sentiment":pred} )
#output.to_csv("Bag_of_Words_model_ETC_best.csv", index=False, quoting=3 )
#!kaggle competitions submit -c word2vec-nlp-tutorial -f Bag_of_Words_model_ETC.csv -m "ETC trying"
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(penalty='l2'
                        , dual=False
                        , tol=0.0001
                        , C=1.0
                        , fit_intercept=True
                        , intercept_scaling=1
                        , class_weight=None
                        , random_state=0
                        , solver='liblinear'
                        , max_iter=100
                        , multi_class='ovr'
                        , verbose=0
                        , warm_start=False
                        , n_jobs=-1)
LR = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.01,
          verbose=0, warm_start=False)
LR.fit(X=X_train_preprocessed_vectorized, y=X_train['sentiment'])
pred = LR.predict(X_test_preprocessed_vectorized)
output = pd.DataFrame( data={"id":X_test["id"], "sentiment":pred} )
output.to_csv("Bag_of_Words_model_LR_best.csv", index=False, quoting=3 )
#cross validation vibes
#X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train['review'].values, X_train['sentiment'].values, test_size=.2)
#%%time
##preprocessiong
#
#num_reviews = X_train_cv.size
#print("Cleaning and parsing the training set movie reviews...\n")
#X_train_cv_preprocessed = []
#for i in iter(range( 0, num_reviews )):
#    if( (i+1)%1000 == 0 ):
#        print("Review %d of %d" % ( i+1, num_reviews ))                                                                 
#    X_train_cv_preprocessed.append( review_to_words( X_train_cv[i] ))
##default vectorizer
##set less dimensions for parameters computation
#
#vectorizer = CountVectorizer(analyzer = "word",
#                             tokenizer = None,
#                             preprocessor = None,
#                             stop_words = stopwords.words("english"),
#                             ngram_range=(1,2),
#                             max_features = 5000)
#
#X_train_cv_preprocessed_vectorized = vectorizer.fit_transform(X_train_cv_preprocessed)
#
#X_train_cv_preprocessed_vectorized = X_train_cv_preprocessed_vectorized.toarray()
#print(X_train_cv_preprocessed_vectorized.shape)
#%%time
##preprocessiong
#
#num_reviews = X_test_cv.size
#print("Cleaning and parsing the training set movie reviews...\n")
#X_test_cv_preprocessed = []
#for i in iter(range( 0, num_reviews )):
#    if( (i+1)%1000 == 0 ):
#        print("Review %d of %d" % ( i+1, num_reviews ))                                                                 
#    X_test_cv_preprocessed.append( review_to_words( X_test_cv[i] ))
##final processing of test dataset
#
#X_test_cv_preprocessed_vectorized = vectorizer.transform(X_test_cv_preprocessed)
#
#X_test_cv_preprocessed_vectorized = X_test_cv_preprocessed_vectorized.toarray()
#print(X_test_cv_preprocessed_vectorized.shape)
#%%time
#%env JOBLIB_TEMP_FOLDER=/tmp
#
##using gpu for grid search cv
#
#params = {
#    'max_iter': [100, 150, 250, 500],
#    'tol': [.0001, .001, .01, 1.0],
#    'C': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
#}
#
#cv = GridSearchCV(estimator=LR, param_grid=params, n_jobs=-1, cv=3, verbose=1)
#cv.fit(X=X_train_cv_preprocessed_vectorized, y=y_train_cv)
#cv.best_estimator_
#from sklearn.metrics import roc_auc_score
#roc_auc_score(y_true=y_test_cv, y_score=cv.best_estimator_.predict(X_test_cv_preprocessed_vectorized))



























