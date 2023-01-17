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
# Model paarameters



BATCH_SIZE = 32

EPOCHS = 400

STEPS_PER_EPOCH = 3000

N_CHARS = 30
import json

import os

import numpy as np

from collections import Counter

from functools import reduce

from keras.models import Model, load_model

from keras.layers import LSTM, Dense, Input

from keras.utils import to_categorical





# all data => 428275 strings

# unique terms => 6714



# We are trying to find capacity of network that will be able to overfit 

# this data





def read_preprocess_data(filename):

    ''' Read file and extract training data'''

    with open(filename,'r') as data_file:

        data = json.load(data_file)

    ingredients = []

    cuisines = []

    for rid_data in data:

        ingredients.extend([e.lower() for e in rid_data['ingredients']])

        cuisines.append(rid_data["cuisine"].lower())

    return ingredients, cuisines





def create_mappings(strings, n_top_chars = 29):

    ''' Construct mappings char -> index and index -> char'''

    cnts = Counter()

    for string in strings:

        cnts.update(string)

    top_chars = [ item[0] for item in cnts.most_common(n=n_top_chars) ]

    char2idx = {char:i for i,char in enumerate(top_chars,1)}

    idx2char = {i:char for i,char in enumerate(top_chars,1)}

    return char2idx, idx2char





def encode_data(strings, char2idx):

    data = []

    for string in strings:

        term = []

        for char in string:

            if char in char2idx:

                term.append(char2idx[char])

            else:

                term.append(0)

        data.append(term)

    return data

    



def data_gen(encoded_data, batch_size, max_len=20, min_len=2, n_chars=30):

    data_arr = encoded_data

    seq_len = np.asarray([len(e) for e in encoded_data])

    while True:

        for seq_min_len in range(min_len, max_len):

            batch_x, batch_y = [],[]

            for idx in np.where(seq_len >=  seq_min_len)[0]:

                batch_x.append( to_categorical(data_arr[idx][:(seq_min_len-1)], n_chars) )

                batch_y.append( to_categorical(data_arr[idx][seq_min_len-1], n_chars) )

                if len(batch_x)==batch_size:

                    yield np.array(batch_x), np.array(batch_y)

                    batch_x, batch_y = [], []
FILE = '../input/train.json'



ingredients, cuisines = read_preprocess_data(FILE)

char2idx, idx2char = create_mappings(ingredients, N_CHARS - 1)

data_encoded = encode_data(ingredients, char2idx)

gen = data_gen(data_encoded, batch_size = BATCH_SIZE, n_chars=N_CHARS)
x = Input(shape = (None,30), name='input_layer')

layer1_x = LSTM(128, activation='tanh', name="layer_1", return_sequences=True)(x)

layer2_x = LSTM(64, activation='tanh', name="layer_2")(layer1_x)

y_hat = Dense(30,activation='softmax' )(layer2_x)

    

model = Model(inputs=x, outputs=y_hat)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit_generator(gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)
def decode_sequence(seq, idx2char):

    output = []

    for el in seq:

        if el in idx2char:

            output.append(idx2char[el])

        else:

            output.append("!")

    return "".join(output)



def ml_model_predictor(model, seq, n_chars=30, next_n_chars=10):

    for i in range(next_n_chars):

        input_x = np.expand_dims( to_categorical(seq, n_chars), axis=0 )

        probs = model.predict(input_x)

        next_char = np.argmax(np.squeeze(probs))

        seq.append(next_char)

    return seq
predicted_sequence = ml_model_predictor(model, data_encoded[0][:4])



print("Prefix  = {0}".format(decode_sequence(data_encoded[0][:4], idx2char)))

print("Predicted  = {0}".format(decode_sequence(predicted_sequence, idx2char)))
model.save("autocomplete_model")
os.listdir(os.getcwd())