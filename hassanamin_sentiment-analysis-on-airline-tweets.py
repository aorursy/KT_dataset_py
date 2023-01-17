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
# Basic packages

import pandas as pd 

import numpy as np

import re

import collections

import matplotlib.pyplot as plt



# Packages for data preparation

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder



# Packages for modeling

from keras import models

from keras import layers

from keras import regularizers
NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

VAL_SIZE = 1000  # Size of the validation set

NB_START_EPOCHS = 20  # Number of epochs we usually start to train with

BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
df = pd.read_csv('../input/Tweets.csv')

df = df.reindex(np.random.permutation(df.index))  

df = df[['text', 'airline_sentiment']]

df.head()
def remove_stopwords(input_text):

        stopwords_list = stopwords.words('english')

        # Some words which might indicate a certain sentiment are kept via a whitelist

        whitelist = ["n't", "not", "no"]

        words = input_text.split() 

        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 

        return " ".join(clean_words) 

    

def remove_mentions(input_text):

        return re.sub(r'@\w+', '', input_text)

       

df.text = df.text.apply(remove_stopwords).apply(remove_mentions)

df.head()
X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)

print('# Train data samples:', X_train.shape[0])

print('# Test data samples:', X_test.shape[0])

assert X_train.shape[0] == y_train.shape[0]

assert X_test.shape[0] == y_test.shape[0]
tk = Tokenizer(num_words=NB_WORDS,

               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',

               lower=True,

               split=" ")

tk.fit_on_texts(X_train)



print('Fitted tokenizer on {} documents'.format(tk.document_count))

print('{} words in dictionary'.format(tk.num_words))

print('Top 5 most common words are:', collections.Counter(tk.word_counts).most_common(5))
X_train_seq = tk.texts_to_sequences(X_train)

X_test_seq = tk.texts_to_sequences(X_test)



print('"{}" is converted into {}'.format(X_train[0], X_train_seq[0]))
def one_hot_seq(seqs, nb_features = NB_WORDS):

    ohs = np.zeros((len(seqs), nb_features))

    for i, s in enumerate(seqs):

        ohs[i, s] = 1.

    return ohs



X_train_oh = one_hot_seq(X_train_seq)

X_test_oh = one_hot_seq(X_test_seq)



print('"{}" is converted into {}'.format(X_train_seq[0], X_train_oh[0]))

print('For this example we have {} features with a value of 1.'.format(X_train_oh[0].sum()))
le = LabelEncoder()

y_train_le = le.fit_transform(y_train)

y_test_le = le.transform(y_test)

y_train_oh = to_categorical(y_train_le)

y_test_oh = to_categorical(y_test_le)



print('"{}" is converted into {}'.format(y_train[0], y_train_le[0]))

print('"{}" is converted into {}'.format(y_train_le[0], y_train_oh[0]))
X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.1, random_state=37)



assert X_valid.shape[0] == y_valid.shape[0]

assert X_train_rest.shape[0] == y_train_rest.shape[0]



print('Shape of validation set:',X_valid.shape)
base_model = models.Sequential()

base_model.add(layers.Dense(64, activation='relu', input_shape=(NB_WORDS,)))

base_model.add(layers.Dense(64, activation='relu'))

base_model.add(layers.Dense(3, activation='softmax'))

base_model.summary()
def deep_model(model):

    model.compile(optimizer='rmsprop'

                  , loss='categorical_crossentropy'

                  , metrics=['accuracy'])

    

    history = model.fit(X_train_rest

                       , y_train_rest

                       , epochs=NB_START_EPOCHS

                       , batch_size=BATCH_SIZE

                       , validation_data=(X_valid, y_valid)

                       , verbose=0)

    

    return history

base_history = deep_model(base_model)
def eval_metric(history, metric_name):

    metric = history.history[metric_name]

    val_metric = history.history['val_' + metric_name]



    e = range(1, NB_START_EPOCHS + 1)



    plt.plot(e, metric, 'bo', label='Train ' + metric_name)

    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)

    plt.legend()

    plt.show()



eval_metric(base_history, 'loss')

eval_metric(base_history, 'acc')