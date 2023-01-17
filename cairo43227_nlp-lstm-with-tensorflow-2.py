# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
missing_cols = ['keyword', 'location']
fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=train[missing_cols].isnull().sum().index, y=train[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x=train[missing_cols].isnull().sum().index, y=train[missing_cols].isnull().sum().values, ax=axes[1])
train['keyword'] = train['keyword'].fillna('no_keyword')
train['location'] = train['location'].fillna('no_location')

test['keyword'] = test['keyword'].fillna('no_keyword')
test['location'] = test['location'].fillna('no_location')
print(f'Number of unique values in keyword = {train["keyword"].nunique()} (Training) - {test["keyword"].nunique()} (Test)')
print(f'Number of unique values in location = {train["location"].nunique()} (Training) - {test["location"].nunique()} (Test)')
fig, axes = plt.subplots( figsize=(8, 4), dpi=100)
plt.tight_layout()
sns.countplot(x=train['target'], hue=train['target'], ax=axes)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import collections
import math
import os
import zipfile
import time
import re
import numpy as np
import tensorflow as tf
from matplotlib import pylab
%matplotlib inline
from six.moves import range
from six.moves.urllib.request import urlretrieve


def text_processing(ft8_text):
    """Replacing punctuation marks with tokens"""
    ft8_text = ft8_text.lower()
    ft8_text = ft8_text.replace('.', ' <period> ')
    ft8_text = ft8_text.replace(',', ' <comma> ')
    ft8_text = ft8_text.replace('"', ' <quotation> ')
    ft8_text = ft8_text.replace(';', ' <semicolon> ')
    ft8_text = ft8_text.replace('!', ' <exclamation> ')
    ft8_text = ft8_text.replace('?', ' <question> ')
    ft8_text = ft8_text.replace('(', ' <paren_l> ')
    ft8_text = ft8_text.replace(')', ' <paren_r> ')
    ft8_text = ft8_text.replace('  ', ' <hyphen> ')
    ft8_text = ft8_text.replace(':', ' <colon> ')
    ft8_text_tokens = ft8_text.split()
    return ft8_text_tokens
# ft_tokens = text_processing(full_text)
"""Shortlisting words with frequency more than 7"""
# word_cnt = collections.Counter(ft_tokens)
# shortlisted_words = [w for w in ft_tokens if word_cnt[w] > 7 ]
def dict_creation(shortlisted_words):
    """The function creates a dictionary of the words present in dataset along with their frequency order"""
    counts = collections.Counter(shortlisted_words)
    vocabulary = sorted(counts, key=counts.get, reverse=True)
    rev_dictionary_ = {ii: word for ii, word in enumerate(vocabulary)}
    dictionary_ = {word: ii for ii, word in rev_dictionary_.items()}
    return dictionary_, rev_dictionary_
# dictionary_, rev_dictionary_ = dict_creation(shortlisted_words)
# words_cnt = [dictionary_[word] for word in shortlisted_words]
X = train.copy(deep=True)
X_test = test.copy(deep=True)

#Create vocabulary

#Sum keywords
X['text_key'] = X['keyword'] + " "+ X['text']
Y = X['target']
X = X.drop(['keyword','id','location', 'text', 'target'], axis=1)

X
vocabulary = []
for words in X['text_key']:
    for word in words.split(' '):
        if word not in vocabulary:
            vocabulary.append(word)
vocabulary
#stop words and punctuation
from nltk.corpus import stopwords
!pip install tensorflow_datasets
import tensorflow_datasets as tfds

stop_words = set(stopwords.words('english'))

vocabulary = [word for word in vocabulary if word not in stop_words]
len(vocabulary)
encoder = tfds.features.text.SubwordTextEncoder(vocabulary)
encoder.vocab_size

# encoded_string = encoder.encode(sample_string)
# print('Encoded string is {}'.format(encoded_string))

# original_string = encoder.decode(encoded_string)
# print('The original string: "{}"'.format(original_string))
code = []
for words in X['text_key']:
    code.append(encoder.encode(words))

X_ = pd.Series(code)    
print(X_)
# train_dataset = (train
#                  .shuffle(BUFFER_SIZE)
#                  .padded_batch(BATCH_SIZE, padded_shapes=([None],[])))

# test_dataset = (test
#                 .padded_batch(BATCH_SIZE,  padded_shapes=([None],[])))
length = max(map(len, X_))
#Tirar esse 0
X_=np.array([xi+[0]*(length-len(xi)) for xi in X_])

X_ = np.asarray(X_)
Y   = np.asarray(Y)
Y = Y.reshape(7613,)
# X_ = np.expand_dims(X_, -1)
# Y   = np.expand_dims(Y, -1)

print("{} - {}".format(X_.shape, Y.shape))
Y= tf.convert_to_tensor(Y)
X_= tf.convert_to_tensor(X_)

X_
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(X_, Y, epochs=10)
#                     validation_data=test_dataset, 
#                     validation_steps=30)


X_test
X_test['text_key'] = X_test['keyword'] + " "+X_test['text']
X = X_test.drop(['keyword','id','location', 'text'], axis=1)

code = []
for words in X_test['text_key']:
    code.append(encoder.encode(words))
X_test = pd.Series(code)    
length = max(map(len, X_test))
#Tirar esse 0
X_test=np.array([xi+[0]*(length-len(xi)) for xi in X_test])
X_test = np.asarray(X_test)
X_test= tf.convert_to_tensor(X_test)


predict = model.predict(X_test)

predict.shape
predict =predict.reshape(3263,)
predict =np.round(predict, 0)
predict1=[]
for i in predict:
    if i==1.:
        predict1.append(1)
    else:
        predict1.append(0)
predict1[:20]
df = pd.DataFrame({'id':test.id, 'target':predict1})
df.to_csv('nlp.csv', index=False)
df.head()