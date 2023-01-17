# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

# load data

train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')

test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t')

train.shape, test.shape
def show_info(data, is_matrix_transpose=False):

    # basic shape

    print('data shape is: {}   sample number {}   attribute number {}\n'.format(data.shape, data.shape[0], data.shape[1]))

    # attribute(key)

    print('data columns number {}  \nall columns: {}\n'.format(len(data.columns) ,data.columns))

    # value's null

    print('data all attribute count null:\n', data.isna().sum())

    # data value analysis and data demo

    if is_matrix_transpose:

        print('data value analysis: ', data.describe().T)

        print('data demo without matrix transpose: ', data.head().T)

    else:

        print('data value analysis: ', data.describe())

        print('data demo without matrix transpose: ', data.head())



show_info(train)

show_info(test)
# plot the sentiment in train

train['Sentiment'].value_counts().plot.bar()
from bs4 import BeautifulSoup

import re

from tqdm import tqdm

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# # split by split() and use stopwords

# def data_preprocessing(df):

#     reviews = []

#     for raw in tqdm(df['Phrase']):

#         # remove html tag

#         text = BeautifulSoup(raw, 'lxml').get_text()

#         # remove non-letters

#         letters_only = re.sub('[^a-zA-Z]', ' ', text)

#         # split(lowercase)

#         words = letters_only.lower().split()

#         # get stoplist words

#         stops = set(stops.words('english'))

#         # remove stopwords / get non-stopwords list

#         non_stopwords = [word for word in words if not word in stops]

#         # lemmatize word to its lemma

#         lemma_words = [lemmatizer.lemmatize(word) for word in words]

#         reviews.append(lemma_words)

#     return reviews





# split by nltk.word_tokenizer

def data_preprocessing(df):

    reviews = []

    for raw in tqdm(df['Phrase']):

        # remove html tag

        text = BeautifulSoup(raw, 'lxml').get_text()

        # remove non-letters

        letters_only = re.sub('[^a-zA-Z]', ' ', text)

        # split(lowercase)

        words = word_tokenize(letters_only.lower())

        # get stoplist words

        stops = set(stopwords.words('english'))

        # remove stopwords / get non-stopwords list

        non_stopwords = [word for word in words if not word in stops]

        # lemmatize word to its lemma

        lemma_words = [lemmatizer.lemmatize(word) for word in non_stopwords]    

        reviews.append(lemma_words)

    return reviews





# data cleaning for train and test

%time train_sentences = data_preprocessing(train)

%time test_sentences = data_preprocessing(test)

len(train_sentences), len(test_sentences)
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
# extract label columns and to_categorical

target = train.Sentiment.values

y_target = to_categorical(target)

num_classes = y_target.shape[1]
# train set => split to train and validation sets

X_train, X_val, y_train, y_val = train_test_split(train_sentences, y_target, test_size=0.2, stratify=y_target)
# keras tokenzier initialization

unique_words = set()

len_max = 0

for sent in tqdm(X_train):

    unique_words.update(sent)

    if len_max < len(sent):

        len_max = len(sent)

len(list(unique_words)), len_max
# transfer to keras tokenizer

tokenizer = Tokenizer(num_words=len(list(unique_words)))

tokenizer.fit_on_texts(list(X_train))



X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)

X_test = tokenizer.texts_to_sequences(test_sentences)



X_train = sequence.pad_sequences(X_train, maxlen=len_max)

X_val = sequence.pad_sequences(X_val, maxlen=len_max)

X_test = sequence.pad_sequences(X_test, maxlen=len_max)



X_train.shape, X_val.shape, X_test.shape
from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.optimizers import Adam
early_stopping = EarlyStopping(min_delta=0.001, mode='max', monitor='val_acc', patience=2)

callback = [early_stopping]
# build model

model = Sequential()

model.add(Embedding(len(list(unique_words)), 300, input_length=len_max))

model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))

model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])

model.summary()
%%time



# fit

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=256, verbose=1, callbacks=callback)
# submit

y_pred = model.predict_classes(X_test)

submission = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')

submission.Sentiment = y_pred

submission.to_csv('submission.csv', index=False)