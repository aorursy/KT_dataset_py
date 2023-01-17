import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Input

from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, GRU, Flatten, concatenate, Embedding, GlobalAveragePooling1D, Conv1D, SpatialDropout1D, BatchNormalization

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model

import re

import seaborn as sns
data_path = '/kaggle/input/sentiment-analysis-for-steam-reviews/train.csv'

data = pd.read_csv(data_path)

data = data[['user_review', 'user_suggestion']]

data.head()
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    #text = text.replace('\%','')

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = " ".join(filter(lambda x:x[0]!="@", text.split()))

    return text



#Apply the function

data['user_review'] = data['user_review'].apply(lambda x: clean_text(x))
data['user_suggestion'].value_counts().plot(kind='bar')
X = data.user_review

y = data.user_suggestion



train_size = int(len(data) * 0.7)

X_train, y_train = X[:train_size], y[:train_size]

X_test, y_test = X[train_size:], y[train_size:]



print("X_train shape: ", X_train.shape)

print("X_test shape: ", X_test.shape)

print("y_train shape: ", y_train.shape)

print("y_test shape: ", y_test.shape)
vocab_size = 10000

max_length = 40

embedding_dim = 16



tokenizer = Tokenizer(vocab_size, oov_token='<OOV>')

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index



sequences = tokenizer.texts_to_sequences(X_train)

train_padded = pad_sequences(sequences, maxlen=max_length, truncating='pre', padding='pre')



test_sequences = tokenizer.texts_to_sequences(X_test)

test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating='pre', padding='pre')



print(train_padded.shape)

print(test_padded.shape)
model = Sequential([

                    Embedding(vocab_size, embedding_dim, input_length=max_length),

                    Bidirectional(LSTM(100, return_sequences=True)),

                    BatchNormalization(),

                    SpatialDropout1D(0.5),

                    Flatten(),

                    Dense(100),

                    BatchNormalization(),

                    Dropout(0.5),

                    Dense(1, activation='sigmoid')

])

model.summary()