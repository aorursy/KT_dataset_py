# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import keras

import json

from datetime import datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
keras.__version__
df = pd.read_csv("../input/movie_conversations.tsv", encoding='utf-8-sig', sep="\t")
df.head()
df = pd.read_csv("../input/movie_lines.tsv", encoding='utf-8-sig',header = None)
lines = df[0].str.split('\t')
dialogue_lines = list()

for x in lines:

    dialogue_lines.append(x[4])

dialogue_lines[:10]
dialogues_path = "../input/movie_lines.tsv"

VOCAB_SIZE = 5000 # len(keras_tokenizer.word_index) + 1

print(VOCAB_SIZE)

EMBEDDING_DIM = 500
len(dialogue_lines)
from keras.preprocessing.text import Tokenizer

from statistics import median

keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}\t\n')

keras_tokenizer.fit_on_texts(dialogue_lines)
keras_tokenizer.word_index
text_sequences = keras_tokenizer.texts_to_sequences(dialogue_lines)[:2000]
MAX_SEQUENCE_LENGTH = int(median(len(sequence) for sequence in text_sequences))

print(MAX_SEQUENCE_LENGTH)
from keras import backend as K

from keras.engine.topology import Layer

from keras.layers import Input, Dense, RepeatVector, LSTM, Conv1D, Masking, Embedding

from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.models import Model

from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', 

                        truncating='post', value=0)
x_train.shape
x_train_rev = list()

for x_vector in x_train:

    x_rev_vector = list()

    for index in x_vector:

        char_vector = np.zeros(VOCAB_SIZE)

        char_vector[index] = 1

        x_rev_vector.append(char_vector)

    x_train_rev.append(np.asarray(x_rev_vector))

x_train_rev = np.asarray(x_train_rev)

x_train_rev.shape
def get_seq2seq_model():

    main_input = Input(shape=x_train[0].shape, dtype='float32', name='main_input')

    print(main_input)



    embed_1 = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, 

                        mask_zero=True, input_length=MAX_SEQUENCE_LENGTH) (main_input)

    print(embed_1)



    lstm_1 = Bidirectional(LSTM(2048, name='lstm_1'))(embed_1)

    print(lstm_1)



    repeat_1 = RepeatVector(MAX_SEQUENCE_LENGTH, name='repeat_1')(lstm_1)

    print(repeat_1)



    lstm_3 = Bidirectional(LSTM(2048, return_sequences=True, name='lstm_3'))(repeat_1)

    print(lstm_3)



    softmax_1 = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(lstm_3)

    print(softmax_1)

    

    model = Model(main_input, softmax_1)

    model.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    

    return model

seq2seq_model = get_seq2seq_model()

seq2seq_model.fit(x_train, x_train_rev, batch_size=128, epochs=20, verbose=1)

predictions = seq2seq_model.predict(x_train)

index2word_map = inv_map = {v: k for k, v in keras_tokenizer.word_index.items()}
def sequence_to_str(sequence):

    word_list = list()

    for element in sequence:

#         if amax(element) < max_prob:

#             continue

        index = np.argmax(element) + 1

        word = index2word_map[index]

        word_list.append(word)

        

    return word_list
#use_eos=True

for i in range(len(predictions)):

        predicted_word_list = sequence_to_str(predictions[i])

        actual_len = len(dialogue_lines[i])



        actual_sentence = "Actual: " + dialogue_lines[i][:len(dialogue_lines[i])-3]        

        

        generated_sentence = ""

        for word in predicted_word_list:

            '''

            if word == EOS_TOKEN:

                predictions_file.write('\n')

                break

            '''

            generated_sentence += word + " "



        sent_dict = dict()

        sent_dict["actual"] = actual_sentence.strip()

        sent_dict["generated"] = generated_sentence.strip()

        print(sent_dict)