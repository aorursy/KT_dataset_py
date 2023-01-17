import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dot, Reshape, GRU, BatchNormalization, Average
from keras import layers, constraints
from keras.preprocessing import sequence
from keras import optimizers
import numpy as np

import tensorflow as tf
import keras.backend as T
from keras import backend as K
from keras import callbacks

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import re
from collections import Counter
train = pd.read_csv('../input/train.tsv', delimiter='\t')
test = pd.read_csv('../input/test.tsv', delimiter='\t')
train['parsed'] = train['Phrase'].apply(lambda x : ' '.join([word.lower() for word in re.split("[ -]+", x)]))
test['parsed'] = test['Phrase'].apply(lambda x : ' '.join([word.lower() for word in re.split("[ -]+", x)]))
def get_features(df):
    features = HashingVectorizer().transform(df['parsed'])
    if 'Sentiment' in df.columns:
        labels = df['Sentiment']
    else:
        labels = None
    return features, labels
kf = KFold(n_splits=5, random_state=None, shuffle=False)

X, y = get_features(train)

for train_index, test_index in kf.split(train):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(metrics.accuracy_score(y_test, LogisticRegression().fit(X_train, y_train).predict(X_test)))
test_X, _ = get_features(test)

submission = pd.DataFrame(list(zip(test['PhraseId'], LogisticRegression().fit(X, y).predict(test_X))), columns=['PhraseId', 'Sentiment'])
submission.to_csv('bow_submission.csv', index=False)
