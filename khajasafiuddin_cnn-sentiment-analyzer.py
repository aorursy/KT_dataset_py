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
import pandas as pd

import numpy as np

import math

import re

from bs4 import BeautifulSoup
from tensorflow.keras import layers

import tensorflow_datasets as tfds

import tensorflow as tf
cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']

data = pd.read_csv('/kaggle/input/tweets-for-sentiment-analysis/training.1600000.processed.noemoticon.csv',

                        header=None,

                        names=cols,

                        engine='python',

                        encoding='latin1')
data.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)
def clean_tweet(tweet):

    tweet = BeautifulSoup(tweet, 'lxml').get_text()

    tweet = re.sub(r'@[A-Za-z0-9]+', ' ', tweet)

    tweet = re.sub(r'https?://[A-Za-z0-9./]+', ' ', tweet)

    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)

    tweet = re.sub(r' +', ' ', tweet)

    return tweet
data_clean = [clean_tweet(tweet) for tweet in data['text']]
data_labels = data['sentiment'].values

data_labels[data_labels == 4] = 1
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(

            data_clean, target_vocab_size=2**16)



data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]
MAX_LEN = max([len(sentence) for sentence in data_inputs])

data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,

                                                           value=0,

                                                           padding = 'post',

                                                           maxlen = MAX_LEN)
test_idx = np.random.randint(0, 800000, 8000)

test_idx = np.concatenate((test_idx, test_idx + 800000))
test_inputs = data_inputs[test_idx]

test_labels = data_labels[test_idx]



train_inputs = np.delete(data_inputs, test_idx, axis=0)

train_labels = np.delete(data_labels, test_idx)
# Building a custom CNN model

class DeepCNN(tf.keras.Model):

    

    def __init__(self, vocab_size, emb_dim=128,

                 nb_filters=50, FFN_units=512,

                 nb_classes=2, dropout_rate=0.1,

                 training=False, name='dcnn'):

        super(DeepCNN, self).__init__(name=name)

        

        self.embedding = layers.Embedding(vocab_size, emb_dim)

        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding='valid', activation='relu')

        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding='valid', activation='relu')

        self.quadgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding='valid', activation='relu')

        

        self.pool = layers.GlobalMaxPool1D()

        

        self.dense_1 = layers.Dense(units=FFN_units, activation='relu')

        self.dropout = layers.Dropout(rate=dropout_rate)

        

        if nb_classes == 2:

            self.last_dense = layers.Dense(units=1, activation='sigmoid')

        

        else:

            self.last_dense = layers.Dense(units=nb_classes, activation='softmax')

    

    def call(self, inputs, training):

        x = self.embedding(inputs)

        

        x_1 = self.bigram(x)

        x_1 = self.pool(x_1)

        

        x_2 = self.trigram(x)

        x_2 = self.pool(x_2)

        

        x_3 = self.quadgram(x)

        x_3 = self.pool(x_3)

        

        merged = tf.concat([x_1, x_2, x_3], axis=-1) # shape: (batch_size, 3*nb_filters)

        

        merged = self.dense_1(merged)

        merged = self.dropout(merged, training)

        

        output = self.last_dense(merged)

        

        return output
# Configuration



VOCAB_SIZE = tokenizer.vocab_size



EMB_DIM = 200

NB_FILTERS = 100

FFN_UNITS = 256

NB_CLASSES = len(set(train_labels))



DROPOUT_RATE = 0.2



BATCH_SIZE = 512

NB_EPOCHS = 5
dcnn = DeepCNN(vocab_size=VOCAB_SIZE,

                emb_dim=EMB_DIM,

                nb_filters=NB_FILTERS,

                FFN_units=FFN_UNITS,

                nb_classes=NB_CLASSES,

                dropout_rate=DROPOUT_RATE,

                )
if NB_CLASSES == 2:

    dcnn.compile(loss='binary_crossentropy',

                optimizer='adam',

                metrics=['accuracy'])

else:

    dcnn.compile(loss='sparse_categorical_crossentropy',

                optimizer='adam',

                metrics=['sparse_categorical_accuracy'])
checkpoint_path = 'ckpt/'

ckpt = tf.train.Checkpoint(dcnn=dcnn)



ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)



if ckpt_manager.latest_checkpoint:

    ckpt.restore(ckpt_manager.latest_checkpoint)

    print('Latest Checkpoint Restored!!')
dcnn.fit(train_inputs,

        train_labels,

        batch_size=BATCH_SIZE,

        epochs=NB_EPOCHS)

ckpt_manager.save()
results = dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)

print(results)
predictions = (dcnn.predict(test_inputs) > 0.5).astype(int)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(test_labels, predictions))

print('\n')

print(classification_report(test_labels, predictions))
dcnn.predict_step(np.array([tokenizer.encode("You ugly ugly") + [0, 0]]))