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
import numpy as np

import pandas as pd

import re

import keras.backend as K

import tensorflow as tf

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model

from keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dropout, Dense, Activation

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model
data = pd.read_csv('/kaggle/input/mcq-distractors/Train.csv')



# only taking sentences of length <= 25

data['question_len'] = data['question'].apply(lambda x: len(x.split()))

data['distractor_len'] = data['distractor'].apply(lambda x: len(x.split()))



data = data[data['question_len']<=25]

data = data[data['distractor_len']<=25]
# remove quotes

data['question'] = data['question'].apply(lambda x: re.sub("'", '', x))

data['distractor'] = data['distractor'].apply(lambda x: re.sub("'", '', x))
source = data.loc[:,'question']

target = data.loc[:,'distractor']



# Adding START_ and _END tokens

target = target.apply(lambda x:'START_ '+x+' _END')

type(target)
#tokenizing the source words

t_source = Tokenizer()

t_source.fit_on_texts(source)

source_vocab_size = len(t_source.word_index)+1

   

#tokenizing the target words

t_target = Tokenizer()

t_target.fit_on_texts(target)

target_vocab_size = len(t_target.word_index)+1
# tokenizing the source and target 

source = t_source.texts_to_sequences(source)

target = t_target.texts_to_sequences(target)

source_train, source_test, target_train, target_test = train_test_split(source, target, test_size=0.2, random_state = 21)

# generating batch for training

def generate_batch(X, y, batch_size=64):

    while True:

        for b in range(0, len(X), batch_size):

            encoder_source = np.zeros([batch_size, 25], dtype='float32')

            decoder_source = np.zeros([batch_size, 26], dtype='float32')

            decoder_target = np.zeros([batch_size, 26, target_vocab_size])

            for i, (source_sen, target_sen) in enumerate(zip(X[b:b+batch_size],y[b:b+batch_size])):

                for j, k in enumerate(source_sen):

                    encoder_source[i, j] = k

                for j, k in enumerate(target_sen):

                    if j<len(target_sen):

                        decoder_source[i, j] = k

                    if j>0:

                        decoder_target[i, j-1, k] = 1

                    

                        

            yield ([encoder_source, decoder_source], decoder_target)

            

            

            
# embedding matrix

embedding_index = dict()

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', encoding="utf8")

for line in f:

   values = line.split()

   word = values[0]

   coefs = np.asarray(values[1:])

   embedding_index[word] = coefs

f.close()

    

embedding_matrix_source = np.zeros([source_vocab_size, 100])

for word, i in t_source.word_index.items():

   embedding_vector = embedding_index.get(word)

   if embedding_vector is not None:

      embedding_matrix_source[i] = embedding_vector

            

embedding_matrix_target = np.zeros([target_vocab_size, 100])

for word, i in t_target.word_index.items():

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_target[i] = embedding_vector
latent_dim = 100

# encoder

encoder_input = Input(shape=(None,))

encoder_embedding = Embedding(source_vocab_size, latent_dim, weights=[embedding_matrix_source], mask_zero=True)

encoder_embedded = encoder_embedding(encoder_input)

encoder_LSTM = LSTM(latent_dim, return_state = True)

source_encoder_outputs, source_state_h, source_state_c = encoder_LSTM(encoder_embedded)

encoder_state = [source_state_h, source_state_c]



# decoder

decoder_input = Input(shape=(None,))

decoder_embedding = Embedding(target_vocab_size, latent_dim, weights=[embedding_matrix_target], mask_zero=True)

decoder_embedded = decoder_embedding(decoder_input)

decoder_LSTM = LSTM(latent_dim, return_state = True, return_sequences=True)

decoder_outputs, _, _ = decoder_LSTM(decoder_embedded, initial_state = encoder_state)

decoder_dense = Dense(target_vocab_size, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



print(model.summary())

model.fit_generator(generator=generate_batch(source_train, target_train, batch_size=64),steps_per_epoch = len(source_train)//64, validation_data=generate_batch(source_test, target_test, batch_size=64), validation_steps=len(source_test)//64, epochs = 25)