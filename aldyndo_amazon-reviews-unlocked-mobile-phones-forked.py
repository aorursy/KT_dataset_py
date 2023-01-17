import re

import nltk



import pandas as pd

import numpy as np



from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

english_stemmer=nltk.stem.SnowballStemmer('english')



from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier, SGDRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import random

import itertools



import sys

import os

import argparse

from sklearn.pipeline import Pipeline

from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer

import six

from abc import ABCMeta

from scipy import sparse

from scipy.sparse import issparse

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils import check_X_y, check_array

from sklearn.utils.extmath import safe_sparse_dot

from sklearn.preprocessing import normalize, binarize, LabelBinarizer

from sklearn.svm import LinearSVC



from keras.preprocessing import sequence

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Lambda

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM, SimpleRNN, GRU

from keras.preprocessing.text import Tokenizer

from collections import defaultdict

from keras.layers.convolutional import Convolution1D

from keras import backend as K



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline

plt.style.use('ggplot')