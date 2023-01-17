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
df = pd.read_csv("../input/movie_lines.tsv", encoding='utf-8-sig',header = None)

df.head()
lines = df[0].str.split('\t')
lines[0:5]
dialogues = list()

for line in lines:

    dialogues.append(line[4])

    

dialogues[0:10]
dialogue_path = "../input/movie_lines.tsv"

vocab_size = 5000

embedding_dim = 500
from keras.preprocessing.text import Tokenizer

from statistics import median

keras_tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}\t\n')

keras_tokenizer.fit_on_texts(dialogues)
len(keras_tokenizer.word_index)
text_sequences = keras_tokenizer.texts_to_sequences(dialogues)[:2000]
max_seq_len = int(median(len(sequence) for sequence in text_sequences))

print(max_seq_len)
from keras import backend as K

from keras.engine.topology import Layer

from keras.layers import Input, Dense, RepeatVector, LSTM, Conv1D, Masking, Embedding

from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.models import Model

from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(text_sequences, maxlen=max_seq_len, padding='post', truncating='post', value=0)
x_train_rev = list()

for x_vector in x_train:

    x_rev_vector = list()

    for index in x_vector:

        char_vector = np.zeros(vocab_size)

        char_vector[index] = 1

        x_rev_vector.append(char_vector)

    x_train_rev.append(np.asarray(x_rev_vector))

x_train_rev = np.asarray(x_train_rev)
x_train_rev.shape
def seq_2_seq_model():

    inputs = Input(shape=x_train[0].shape, dtype='int32', name='input_layer')

    embedding = Embedding(

        input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, input_length=max_seq_len)(inputs)

    bi_lstm_1 = Bidirectional(LSTM(units=2048, name='bi_lstm_1'))(embedding)

    repeat = RepeatVector(max_seq_len, name='repeat_vector')(bi_lstm_1)

    bi_lstm_3 = Bidirectional(LSTM(units=2048, return_sequences=True, name='bi_lstm_3'))(repeat)

    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(bi_lstm_3)

    

    model = Model(inputs, output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
seq_model = seq_2_seq_model()
seq_model.fit(x_train, x_train_rev, batch_size=128, epochs=20, verbose=1)