

from __future__ import print_function



import logging

import numpy as np

from optparse import OptionParser

import sys

from time import time

import matplotlib.pyplot as plt



from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import RidgeClassifier

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestCentroid

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.extmath import density

from sklearn import metrics

from sklearn.model_selection import train_test_split

import pandas as pd

import eli5



import re

from tqdm import *





def cleaner(word):

  word = re.sub(r'\#\.', '', word)

  word = re.sub(r'\n', '', word)

  word = re.sub(r',', '', word)

  word = re.sub(r'\-', ' ', word)

  word = re.sub(r'\.', '', word)

  word = re.sub(r'\\', ' ', word)

  word = re.sub(r'\\x\.+', '', word)

  word = re.sub(r'\d', '', word)

  word = re.sub(r'^_.', '', word)

  word = re.sub(r'_', ' ', word)

  word = re.sub(r'^ ', '', word)

  word = re.sub(r' $', '', word)

  word = re.sub(r'\?', '', word)



  return word.lower()





def hashing(word):

  word = re.sub(r'ain$', r'ein', word)

  word = re.sub(r'ai', r'ae', word)

  word = re.sub(r'ay$', r'e', word)

  word = re.sub(r'ey$', r'e', word)

  word = re.sub(r'ie$', r'y', word)

  word = re.sub(r'^es', r'is', word)

  word = re.sub(r'a+', r'a', word)

  word = re.sub(r'j+', r'j', word)

  word = re.sub(r'd+', r'd', word)

  word = re.sub(r'u', r'o', word)

  word = re.sub(r'o+', r'o', word)

  word = re.sub(r'ee+', r'i', word)

  if not re.match(r'ar', word):

    word = re.sub(r'ar', r'r', word)

  word = re.sub(r'iy+', r'i', word)

  word = re.sub(r'ih+', r'eh', word)

  word = re.sub(r's+', r's', word)

  if re.search(r'[rst]y', 'word') and word[-1] != 'y':

    word = re.sub(r'y', r'i', word)

  if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):

    word = re.sub(r'i$', r'y', word)

  if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):

    word = re.sub(r'h', '', word)

  word = re.sub(r'k', r'q', word)

  return word



def array_cleaner(array):

  # X = array

  X = []

  for sentence in array:

    clean_sentence = ''

    words = str(sentence).split(' ')

    for word in words:

      clean_sentence = clean_sentence +' '+ cleaner(word)

    X.append(clean_sentence)

  return X

data = pd.read_csv('../input/roman-urdu-dataset/Dataset/Roman Urdu DataSet.csv', encoding="ISO-8859-1", header=None)

data.head()
numpy_array = data.as_matrix()

X = numpy_array[:, 0]

# Clean X here

X_train = array_cleaner(X)

y_train = numpy_array[:, 1]
ngram = 3

vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.5)

X_train = vectorizer.fit_transform(X_train)

def benchmark(clf, name):

  print('_' * 80)

  print("Training: ")

  print(clf)

  clf.fit(X_train, y_train)

  return clf
# clf = benchmark(RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier")

clf = benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"), 'SGD-elasticnet')

# clf = benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty='l1'), 'SGD-L1')

# clf = benchmark(LinearSVC(penalty='l1', dual=False,tol=1e-3), 'liblinear L1')

# clf = benchmark(LinearSVC(penalty='l2', dual=False,tol=1e-3), 'liblinear L2')

# clf = benchmark(MultinomialNB(alpha=.01), 'MultiNB')

# clf = benchmark(BernoulliNB(alpha=.01), 'BernoulliNB')

# clf = benchmark(NearestCentroid(), 'Rocchio')

# clf = benchmark(KNeighborsClassifier(n_neighbors=10), "kNN")

# clf = benchmark(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")



eli5.show_weights(clf, vec=vectorizer)
test_sentence = "Movie achi thi magar hero bura tha"

eli5.show_prediction(clf, doc=test_sentence, vec=vectorizer)