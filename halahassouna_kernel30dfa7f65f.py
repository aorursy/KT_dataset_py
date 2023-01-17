# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import bz2

import gc

import chardet

import re

! pip install tensorflow==2.0.0-rc0
import tensorflow as tf

tf.__version__
from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPool1D, Dropout, concatenate, Layer, InputSpec, CuDNNLSTM

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras import activations, initializers, regularizers, constraints

from keras.utils.conv_utils import conv_output_length

from keras.regularizers import l2

from keras.constraints import maxnorm
train_file = bz2.BZ2File('../input/amazonreviews/train.ft.txt.bz2')

test_file = bz2.BZ2File('../input/amazonreviews/test.ft.txt.bz2')
train_file_lines = train_file.readlines()

test_file_lines = test_file.readlines()
train_file_lines
train_file_lines = [x.decode('utf-8') for x in train_file_lines]

test_file_lines = [x.decode('utf-8') for x in test_file_lines]
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]

train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]



for i in range(len(train_sentences)):

    train_sentences[i] = re.sub('\d','0',train_sentences[i])

    

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]

test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]



for i in range(len(test_sentences)):

    test_sentences[i] = re.sub('\d','0',test_sentences[i])

                                                       

for i in range(len(train_sentences)):

    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:

        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

        

for i in range(len(test_sentences)):

    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:

        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])
train_labels
from keras.preprocessing import text, sequence



max_features = 20000

maxlen = 100

tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(train_sentences)

tokenized_train = tokenizer.texts_to_sequences(train_sentences)

X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(test_sentences)

X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
from tensorflow import keras

from tensorflow.keras import layers

# import tensorflow_datasets as tfds

# tfds.disable_progress_bar()
embedding_layer = layers.Embedding(1000, 5)

import numpy

result = embedding_layer(tf.constant([1,2,3]))

result.numpy()
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))

result.shape
from keras.preprocessing.text import text_to_word_sequence

# define the document

text = 'The quick brown fox jumped over the lazy dog.'



# tokenize the document

result = text_to_word_sequence(text)

print(result)
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

MAX_FEATURES = 12000

tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(train_ds)

train_texts = tokenizer.texts_to_sequences(train_ds)

val_texts = tokenizer.texts_to_sequences(val_ds)

test_texts = tokenizer.texts_to_sequences(test_ds)
encoder = val_ds.features['text'].encoder

encoder.subwords[:20]
padded_shapes = ([None],())

train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)

test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
embedding_dim=16



model = keras.Sequential([

  layers.Embedding(encoder.vocab_size, embedding_dim),

  layers.GlobalAveragePooling1D(),

  layers.Dense(1, activation='sigmoid')

])



model.summary()
train_data = tf.data.Dataset.from_tensor_slices(

    (x_train, y_train)).shuffle(10000).batch(32)



test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)