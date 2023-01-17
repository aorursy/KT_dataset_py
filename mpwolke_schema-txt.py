# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

tf.__version__
path_to_file = '../input/original-json-schema/json_schema.txt'
text = open(path_to_file, 'r',encoding='utf-8',

                 errors='ignore').read()
print(text[:1000])
# The unique characters in the file

vocab = sorted(set(text))

print(vocab)

len(vocab)
char_to_ind = {u:i for i, u in enumerate(vocab)}

ind_to_char = np.array(vocab)

encoded_text = np.array([char_to_ind[c] for c in text])

seq_len = 250

total_num_seq = len(text)//(seq_len+1)

total_num_seq
# Create Training Sequences

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)



sequences = char_dataset.batch(seq_len+1, drop_remainder=True)



def create_seq_targets(seq):

    input_txt = seq[:-1]

    target_txt = seq[1:]

    return input_txt, target_txt



dataset = sequences.map(create_seq_targets)
# Batch size

batch_size = 128



# Buffer size to shuffle the dataset so it doesn't attempt to shuffle

# the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements

buffer_size = 10000



dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)



# Length of the vocabulary in chars

vocab_size = len(vocab)



# The embedding dimension

embed_dim = 64



# Number of RNN units

rnn_neurons = 2052
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU

from tensorflow.keras.losses import sparse_categorical_crossentropy
def sparse_cat_loss(y_true,y_pred):

    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):

    model = Sequential()

    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))

    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))

    # Final Dense Layer to Predict

    model.add(Dense(vocab_size))

    model.compile(optimizer='adam', loss=sparse_cat_loss) 

    return model
model = create_model(

  vocab_size = vocab_size,

  embed_dim=embed_dim,

  rnn_neurons=rnn_neurons,

  batch_size=batch_size)
model.summary()
epochs = 3 