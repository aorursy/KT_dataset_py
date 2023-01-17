import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow_hub as hub



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.metrics  import classification_report, confusion_matrix



import os

os.listdir("../input")
train = pd.read_csv('../input/train.csv', sep=',')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
sns.countplot(train['target']);
t1 = train[train['target']==1]

t0 = train[train['target']==0][:99999]



del train
train = t1.append(t0)

train.head()
train['length']  = train['question_text'].map(len)
train['length'].describe()
import nltk, re, string

from keras.preprocessing import sequence, text

def text_preprocess(text):

    text = text.lower()

    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)

    return text
train['clean_text'] = train['question_text'].map(text_preprocess)

test['clean_text'] = test['question_text'].map(text_preprocess)



train['clean_text'].head()
max_features = 2000

tokenizer = text.Tokenizer(num_words=max_features, lower=True)
train['clean_text'][:10]
tokenizer.fit_on_texts(train['clean_text'])
X_train = tokenizer.texts_to_sequences(train['clean_text'])
X_test = tokenizer.texts_to_sequences(test['clean_text'])
def encode(x):

    return keras.utils.to_categorical(x)



def decode(x):

    return np.argmax(x, axis=1)
x_train = list(X_train)

y_train = list(train['target'])
y_train = encode(y_train)

y_train
from keras.models import Sequential, Model

from keras.layers import Dense, Activation, Dropout, Lambda, Input, Embedding, Bidirectional, LSTM

import keras.backend as K



from keras.preprocessing.text import text_to_word_sequence

import tensorflow as tf
max_features = 2196
model = Sequential()

hidden_size = 128

# model.add(Dense(activation='relu'))

model.add(Embedding(max_features, hidden_size))

model.add(Bidirectional(LSTM(hidden_size, activation='elu', dropout=0.4, recurrent_dropout=0.2)))

model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
x_train = sequence.pad_sequences(X_train, maxlen=71, padding='post', truncating='post')
model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, validation_split=0.1)
score = model.evaluate(x=x_train[25:10025], y=y_train[25:10025], batch_size=16)
score
# y_pr = decode(y_preds)
# c = confusion_matrix(np.argmax(y_train, axis=1), y_pr)

# sns.heatmap(c, annot=True);
x_test = list(X_test)
x_test = sequence.pad_sequences(x_test, maxlen=71, padding='post', truncating='post')
test_preds = model.predict(x_test, batch_size=64)
test_preds = np.argmax(test_preds, axis=1)

test_preds
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
sub['prediction'] = test_preds
sns.countplot(test_preds);
sub.to_csv('submission.csv', index=False)