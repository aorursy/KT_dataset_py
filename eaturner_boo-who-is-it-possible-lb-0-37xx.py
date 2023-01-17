# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
word = []



for text in train['text']:

    word.append( text )



for text in test['text']:

    word.append( text )
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



count_vec = CountVectorizer( ngram_range = (1, 3) )

tfid_ = TfidfTransformer( )



print('Extracing Count Information')

count_vec.fit(word)

train_sparse = count_vec.transform( train['text'] )

test_sparse = count_vec.transform( test['text'] )



print('Normalizing Count Information')

tfid_.fit( train_sparse )



train_tfid = tfid_.transform( train_sparse )

test_tfid = tfid_.transform( test_sparse )
author_dict = { 'EAP' : 0, 'HPL' : 1, 'MWS' : 2 }



author_labels = train['author'].apply(author_dict.get)

train = train.drop('author', axis = 1)

train.drop('id', axis = 1, inplace = True)
test_preds = {}
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(loss = 'log', max_iter = 2000, n_jobs = -1)



sgd_clf.fit( train_tfid, author_labels )



test_preds['sgd_clf'] = sgd_clf.predict_proba( test_tfid )
from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB( )



nb.fit( train_tfid, author_labels )



test_preds['nb_clf'] = nb.predict_proba( test_tfid )
from sklearn.linear_model import LogisticRegression



log_clf = LogisticRegression( solver = 'saga', multi_class = 'multinomial', 

                             max_iter = 500, n_jobs = -1)



log_clf.fit( train_tfid, author_labels )



test_preds['log_clf'] = log_clf.predict_proba( test_tfid )
cols = ['EAP', 'HPL', 'MWS']

sub[cols] = 0.0



n = len( test_preds.keys() )



for key in test_preds.keys():

    sub[cols] += (1.0/n)*( test_preds.get(key) ** -1.0)

    

sub[cols] = ( sub[cols].values ) ** -1.0
sub.to_csv('sub.csv', index = False)