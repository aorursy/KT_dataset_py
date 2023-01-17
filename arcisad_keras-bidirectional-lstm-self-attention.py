# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras.preprocessing.text import one_hot, Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import *

from keras.layers.embeddings import Embedding

from keras.models import Sequential

import pyprind
# https://pypi.org/project/keras-self-attention/

import sys

sys.path.insert(0, '../input/attention')

from seq_self_attention import SeqSelfAttention
df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
trainig_sample = df.sample(100000, random_state=0)

X_train = trainig_sample['comment_text'].astype(str)

X_train = X_train.fillna('DUMMY')

y_train = trainig_sample['target']

y_train = y_train.apply(lambda x: 1 if x > 0.5 else 0)
max_num_words = 20000

max_length = 128

tokenizer = Tokenizer(num_words=max_num_words)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
def get_seqs(text):

    sequences = tokenizer.texts_to_sequences(text)

    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences
model = Sequential()

model.add(Embedding(max_num_words, 100, input_length=max_length))

model.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))

model.add(SeqSelfAttention(attention_activation='sigmoid'))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(get_seqs(X_train), y_train, epochs=2)
validation_sample = df.sample(500, random_state=42)

X_val = validation_sample['comment_text'].astype(str)

X_val = X_val.fillna('DUMMY')

y_val = validation_sample['target']

y_val = y_val.apply(lambda x: 1 if x > 0.5 else 0)
loss, accuracy = model.evaluate(get_seqs(X_val), y_val)

print('Evaluation accuracy: {0}'.format(accuracy))
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
X_test = test['comment_text'].astype(str)

X_test = X_test.fillna('DUMMY')
probs = model.predict(get_seqs(X_test), verbose=1)
probs = [x[0] for x in probs]
submission = pd.DataFrame(test['id']).reset_index(drop=True)

submission['prediction'] = pd.Series(probs, name='prediction')

submission.to_csv('submission.csv', index=False)