import pickle

import pandas as pd

import numpy as np

import string

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2

import re

import gzip
#load the pickle files for train&test sets

with open("../input/text-classification-2-feature-engineering/df_train.pkl", 'rb') as data:

    df_train = pickle.load(data)

    

with open("../input/text-classification-2-feature-engineering/df_test.pkl", 'rb') as data:

    df_test = pickle.load(data)
#since binary=True, it will generate a vector for each word with binary values(like 0-1:presence-absence)



one_hot = CountVectorizer(binary=True, lowercase=False, max_features=1000, ngram_range=(1,2))



X_train_1hot=one_hot.fit_transform(df_train['review_parsed']).toarray()

X_test_1hot=one_hot.transform(df_test['review_parsed']).toarray()

#here since binary=False, it will generate a vector for each word with numeric values(how many of this word review?)



bow = CountVectorizer(binary=False, lowercase=False, max_features=1000, ngram_range=(1,2))



X_train_bow=bow.fit_transform(df_train['review_parsed']).toarray()

X_test_bow=bow.transform(df_test['review_parsed']).toarray()

#Now, let's dump some pickles since we'll use them in later work



# x_train_1hot

with gzip.open('x_train_1hot.pkl', 'wb') as output:

    pickle.dump(X_train_1hot, output, protocol=-1)

    

    

# x_test_1hot    

with gzip.open('x_test_1hot.pkl', 'wb') as output:

    pickle.dump(X_test_1hot, output, protocol=-1)

    



# x_train_bow

with gzip.open('x_train_bow.pkl', 'wb') as output:

    pickle.dump(X_train_bow, output, protocol=-1)

    

    

# x_test_bow    

with gzip.open('x_test_bow.pkl', 'wb') as output:

    pickle.dump(X_test_bow, output, protocol=-1) 

    

    

# one_hot

with gzip.open('one_hot.pkl', 'wb') as output:

    pickle.dump(one_hot, output, protocol=-1)

    

    

# bow

with gzip.open('bow.pkl', 'wb') as output:

    pickle.dump(bow, output, protocol=-1)

    