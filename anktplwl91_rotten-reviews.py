# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D, GRU, Bidirectional, GlobalAveragePooling1D, Conv1D
import keras.backend as K
from keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
train_df.head()
test_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
test_df.head()
all_df = train_df.append(test_df)
phrase_list = all_df['Phrase'].tolist()
keras_tok = text.Tokenizer()
keras_tok.fit_on_texts(phrase_list)
all_phrases = keras_tok.texts_to_sequences(phrase_list)
X = sequence.pad_sequences(all_phrases, 100)
X_train = X[:train_df.shape[0], :]
X_test = X[train_df.shape[0]:, :]
max_features = len(keras_tok.word_counts)
embed_size = 100
le = LabelEncoder()
one_hot = OneHotEncoder(sparse=False)
labels = le.fit_transform(train_df['Sentiment'].values)
y_onehot = one_hot.fit_transform(labels.reshape((-1, 1)))
y_onehot.shape
bi_lstm_conv_model = Sequential()
bi_lstm_conv_model.add(Embedding(max_features+1, embed_size, input_length=100))
bi_lstm_conv_model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.2)))
bi_lstm_conv_model.add(Conv1D(32, 3, activation='relu'))
bi_lstm_conv_model.add(GlobalAveragePooling1D())
bi_lstm_conv_model.add(Dense(5, activation='softmax'))

bi_lstm_conv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
bi_lstm_conv_model.fit(X_train, y_onehot, batch_size=64, epochs=2)
preds = bi_lstm_conv_model.predict(X_test, batch_size=64)
preds = preds.argmax(axis=1)
subm_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
subm_df['Sentiment'] = preds
subm_df.to_csv('bilstm_conv.csv', index=False)
