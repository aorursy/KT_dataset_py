import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from nltk.tokenize import TweetTokenizer

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

pd.set_option('max_colwidth',400)

pd.set_option('max_columns', 50)

import json

import altair as alt

from  altair.vega import v3

from IPython.display import HTML

import gc

import os

print(os.listdir("../input"))

import lime

import eli5

from eli5.lime import TextExplainer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train.head()
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

for col in identity_columns + ['target']:

    train[col] = np.where(train[col] >= 0.2, True, False)

train_x = train[identity_columns]

train_y = train['target']
train_x,train_y
%%time

tokenizer = TweetTokenizer()



vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize, max_features=30000)

vectorizer.fit(train['comment_text'].values)

train_vectorized = vectorizer.transform(train['comment_text'].values)
print(train_vectorized)
%%time

logreg = LogisticRegression()

logreg.fit(train_vectorized, train_y)
%%time

tokenizer = TweetTokenizer()



vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize, max_features=30000)

vectorizer.fit(test['comment_text'].values)

train_vectorized = vectorizer.transform(test['comment_text'].values)
submission = logreg.predict_proba(train_vectorized)[:, 1]
sub['prediction'] = submission
sub
sub.to_csv('submission.csv',index=False)