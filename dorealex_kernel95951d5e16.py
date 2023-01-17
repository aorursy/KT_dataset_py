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
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import pandas as pd

import numpy as np

df = pd.read_csv("../input/atip-data/Open_gov_atip_nil_responses.csv",usecols =['summary_en'])
df



#https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/
df= df.dropna()

tx = df.summary_en.values
tx
n_messages = len(tx)

n_chars = len(" ".join(map(str,tx)))

tx = ' '.join(map(str, tx)).lower()
chars = sorted(list(set(tx)))

print('Count of unique characters (i.e., features):', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40

step = 3

sentences = []

next_chars = []

for i in range(0, len(tx) - maxlen, step):

    sentences.append(tx[i: i + maxlen])

    next_chars.append(tx[i + maxlen])

print('Number of sequences:', len(sentences), "\n")



print(sentences[:10], "\n")

print(next_chars[:10])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):

    for t, char in enumerate(sentence):

        x[i, t, char_indices[char]] = 1

    y[i, char_indices[next_chars[i]]] = 1
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.layers import LSTM

from keras.optimizers import RMSprop

from keras.callbacks import LambdaCallback, ModelCheckpoint

import random

import sys

import io
model = Sequential()

model.add(LSTM(128, input_shape=(maxlen, len(chars))))

model.add(Dense(len(chars)))

model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
def sample(preds, temperature=1.0):

    # helper function to sample an index from a probability array

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)



def on_epoch_end(epoch, logs):

    # Function invoked for specified epochs. Prints generated text.

    # Using epoch+1 to be consistent with the training epochs printed by Keras

    if epoch+1 == 1 or epoch+1 == 15:

        print()

        print('----- Generating text after Epoch: %d' % epoch)



        start_index = random.randint(0, len(tx) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:

            print('----- diversity:', diversity)



            generated = ''

            sentence = tx[start_index: start_index + maxlen]

            generated += sentence

            print('----- Generating with seed: "' + sentence + '"')

            sys.stdout.write(generated)



            for i in range(400):

                x_pred = np.zeros((1, maxlen, len(chars)))

                for t, char in enumerate(sentence):

                    x_pred[0, t, char_indices[char]] = 1.



                preds = model.predict(x_pred, verbose=0)[0]

                next_index = sample(preds, diversity)

                next_char = indices_char[next_index]



                generated += next_char

                sentence = sentence[1:] + next_char



                sys.stdout.write(next_char)

                sys.stdout.flush()

            print()

    else:

        print()

        print('----- Not generating text after Epoch: %d' % epoch)



generate_text = LambdaCallback(on_epoch_end=on_epoch_end)
filepath = "weights.hdf5"

checkpoint = ModelCheckpoint(filepath, 

                             monitor='loss', 

                             verbose=1, 

                             save_best_only=True, 

                             mode='min')



# fit model using our gpu

with tf.device('/gpu:0'):

    model.fit(x, y,

              batch_size=128,

              epochs=15,

              verbose=2,

              callbacks=[generate_text, checkpoint])