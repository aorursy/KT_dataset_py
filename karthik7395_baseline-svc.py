# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import operator
import spacy
import re
import string
from nltk.util import ngrams
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')
train_df.shape
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3),
                        strip_accents='unicode',
                        lowercase =True, analyzer='word',
                        use_idf=True, smooth_idf=True, sublinear_tf=True, 
                        stop_words = 'english',tokenizer=word_tokenize)
vectorizer.fit(train_df.question_text.values)
train_vectorized = vectorizer.transform(train_df.question_text.values)
test_df=pd.read_csv('../input/test.csv')
test_df.head()
test_vectorized = vectorizer.transform(test_df.question_text.values)
X_train, X_val, y_train, y_val=train_test_split(train_vectorized,train_df.target.values,test_size=0.1,stratify =train_df.target.values)
from sklearn.svm import LinearSVC
svc = LinearSVC(dual=True,C=5,penalty='l2',max_iter=1000,tol=0.01)
svc.fit(X_train, y_train)
svc_preds=svc.predict(X_train)
from sklearn.metrics import f1_score, precision_score,recall_score
print(f1_score(y_train,svc_preds))
print(precision_score(y_train,svc_preds))
print(recall_score(y_train,svc_preds))
svc_val_preds=svc.predict(X_val)
from sklearn.metrics import f1_score, precision_score,recall_score
print(f1_score(y_val,svc_val_preds))
print(precision_score(y_val,svc_val_preds))
print(recall_score(y_val,svc_val_preds))
svc_test_preds=svc.predict(test_vectorized)
sample_sub=pd.read_csv('../input/sample_submission.csv')
sample_sub.prediction=svc_test_preds
sample_sub.to_csv('submission.csv',index=False)
