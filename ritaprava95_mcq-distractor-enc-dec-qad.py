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
# import packages

import os

import sys

import numpy as np

import pandas as pd

import string

import re

import tensorflow as tf

import keras.backend as K

from keras.preprocessing.text import Tokenizer

from keras.layers import Input, Embedding, SimpleRNN, GRU, LSTM, Concatenate, Bidirectional, Dense, Activation

from keras.models import Model

from sklearn.model_selection import train_test_split
data = pd.read_csv('/kaggle/input/mcq-distractors/Train.csv').iloc[:,:3]

print(data.head())
# drop duplicates



data.drop_duplicates(inplace=True)
# remove quotes



data['question'] = data['question'].apply(lambda x: re.sub("'", '', x))

data['answer_text'] = data['answer_text'].apply(lambda x: re.sub("'", '', x))

data['distractor'] = data['distractor'].apply(lambda x: re.sub("'", '', x))

print(data.head())
# filter long texts



data['question_len'] = data['question'].apply(lambda x: len(x.split()))

data['answer_len'] = data['answer_text'].apply(lambda x: len(x.split()))

data['distractor_len'] = data['distractor'].apply(lambda x: len(x.split()))

print(len(data))

data = data[data['question_len']<=25]

data = data[data['answer_len']<=25]

data = data[data['distractor_len']<=25]

print(len(data))



# max question and answer length

max_question_len = 25

max_ans_len = 25 

max_dis_len = 26 # i will add START_ and _END # decoder input will have the START_ tokens and decoder output will have the _END
# convert to lower 



data['question'] = data['question'].apply(lambda x: x.lower())

data['answer_text'] = data['answer_text'].apply(lambda x: x.lower())

data['distractor'] = data['distractor'].apply(lambda x: x.lower())

print(data.head())
# exclude special characters



special_char = string.punctuation

data['question'] = data['question'].apply(lambda x: ''.join(ch for ch in x if ch not in special_char))

data['answer_text'] = data['answer_text'].apply(lambda x: ''.join(ch for ch in x if ch not in special_char))

data['distractor'] = data['distractor'].apply(lambda x: ''.join(ch for ch in x if ch not in special_char))

print(data.head())
# add START_ and _END tokens



data['distractor'] = data['distractor'].apply(lambda x: 'START_ '+x+' _END')

print(data.head())
# creating question and answer vocabulary

all_question_words = []

for i in data['question']:

    all_question_words += i.split()



all_question_words = list(set(all_question_words))



all_ans_words = []

for i in data['answer_text']:

    all_ans_words += i.split()

    

all_ans_words = list(set(all_ans_words))



all_dis_words = []

for i in data['distractor']:

    all_dis_words += i.split()

    

all_dis_words = list(set(all_dis_words))
source_words = sorted(list(set(all_question_words +all_ans_words)))

target_words = sorted(all_dis_words)



num_source_tokens = len(source_words)

num_target_tokens = len(target_words)

# tokenizing words



source_tokens = dict([(word, i+1) for i, word in enumerate(source_words)])

target_tokens = dict([(word, i+1) for i, word in enumerate(target_words)])
# reverse tokenizing



reverse_source_tokens = dict([(i,word) for word, i in source_tokens.items()])

reverse_target_tokens = dict([(i,word) for word, i in target_tokens.items()])

# train and test data 



source1, source2, target = data['question'], data['answer_text'], data['distractor']

source1_train, source1_test, source2_train, source2_test, target_train, target_test = train_test_split(source1, source2, target, test_size=0.2, random_state=21)
# function to generate batches as the entire data is to large to store at once. 

def generate_batch(X1, X2, y, batch_size=64):

    while True:

        for b in range(0, len(X1), batch_size):

            encoder_source1_train = np.zeros((batch_size, max_question_len), dtype='float32')

            encoder_source2_train = np.zeros((batch_size, max_ans_len), dtype='float32')

            decoder_source_train = np.zeros((batch_size, max_dis_len), dtype='float32')

            decoder_target_train = np.zeros((batch_size, max_dis_len, num_target_tokens))

            for i, (source1_text, source2_text, target_text) in enumerate(zip(X1[b:b+batch_size], X2[b:b+batch_size], y[b:b+batch_size])):

                for t, word in enumerate(source1_text.split()):

                    encoder_source1_train[i, t] = source_tokens[word]

                for t, word in enumerate(source2_text.split()):

                    encoder_source2_train[i, t] = source_tokens[word]

                for t, word in enumerate(target_text.split()):

                    if t<len(target_text.split())-1:

                        decoder_source_train[i, t] = target_tokens[word]

                    if t>0:

                        decoder_target_train[i, t-1, target_tokens[word]-1] = 1

            yield ([encoder_source1_train, encoder_source2_train, decoder_source_train], decoder_target_train)

 
latent_dim = 300



# encoder

encoder_input1 = Input(shape=(None,))

encoder_input2 = Input(shape=(None,))

encoder_embedding = Embedding(num_source_tokens, latent_dim, mask_zero=True)

source1_encoded_embedding = encoder_embedding(encoder_input1)

source2_encoded_embedding = encoder_embedding(encoder_input2)

encoder_LSTM = LSTM(latent_dim, return_state = True)

source1_encoder_outputs, source1_state_h, source1_state_c = encoder_LSTM(source1_encoded_embedding)

source2_encoder_outputs, source2_state_h, source2_state_c = encoder_LSTM(source2_encoded_embedding)

source_h_state = Concatenate(axis=-1)([source1_state_h, source2_state_h])

source_c_state = Concatenate(axis=-1)([source1_state_c, source2_state_c])

encoder_state = [source_h_state, source_c_state]
# decoder



decoder_input = Input(shape=(None,))

decoder_embedding = Embedding(num_target_tokens, latent_dim, mask_zero=True)

decoded_embedding = decoder_embedding(decoder_input)

decoder_LSTM = LSTM(600, return_state=True, return_sequences=True)

decoder_output, _, _ = decoder_LSTM(decoded_embedding, initial_state = encoder_state)

decoder_dense = Dense(num_target_tokens, activation='softmax')

decoder_output = decoder_dense(decoder_output)
model = Model([encoder_input1, encoder_input2, decoder_input], decoder_output)

print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#traing model

model.fit_generator(generator=generate_batch(source1_train, source2_train, target_train, batch_size=64),steps_per_epoch = len(source1_train)//64, validation_data=generate_batch(source1_test, source2_test, target_test, batch_size=64), validation_steps=len(source1_test)//64, epochs = 25)
len(target_tokens)