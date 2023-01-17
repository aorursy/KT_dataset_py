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
from __future__ import print_function

from keras.callbacks import LambdaCallback

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.optimizers import RMSprop

from keras.optimizers import Adam

from keras.utils.data_utils import get_file

import numpy as np

import random

import sys

import io



df = pd.read_csv('../input/Donald-Tweets!.csv')

print(df.shape)

df.head()
# lowercase all

text = df['Tweet_Text'].str.lower()

np.random.seed(12345)

np.random.choice(text,10)
print('BEFORE:')

print(text[0])

text = text.map(lambda s: ' '.join([x for x in s.split() if 'http' not in x]))

print('AFTER:')

print(text[0])
print('max tweet len:',text.map(len).max())

print('min tweet len:',text.map(len).min())

text.map(len).hist();

text = text[text.map(len)>60]

len(text)
chars = sorted(list(set(''.join(text))))

print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))

chars
for c in chars[-19:]:

    print('\nCHAR:', c)

    smple = [x for x in text if c in x]

    print(random.sample(smple,min(3,len(smple))))
import re

for c in chars[-19:]:

    text = text.str.replace(c,'')
chars = sorted(list(set(''.join(text))))

print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))



chars
# cut the text in semi-redundant sequences of maxlen characters

maxlen = 40

step = 1

sentences = []

next_chars = []

for x in text:

    for i in range(0, len(x) - maxlen, step):

        sentences.append(x[i: i + maxlen])

        next_chars.append(x[i + maxlen])

print('nb sequences:', len(sentences))

## check example

for i in range(3):

    print(sentences[i],'==>',next_chars[i])
text[0]
print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):

    for t, char in enumerate(sentence):

        x[i, t, char_indices[char]] = 1

    y[i, char_indices[next_chars[i]]] = 1

print('Build model...')

model = Sequential()

model.add(LSTM(128, input_shape=(maxlen, len(chars))))

model.add(Dense(len(chars), activation='softmax'))



# optimizer = RMSprop(lr=0.01)

optimizer = Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):

    # helper function to sample an index from a probability array

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

for temperature in [0.1, 0.2, 0.3,  0.5, 1.0, 1.2, 1.3]:

    print(sample([.1,.3,.5,.1],temperature=temperature))
def on_epoch_end(epoch, _):

    # Function invoked at end of each epoch. Prints generated text.

    print()

    print('----- Generating text after Epoch: %d' % epoch)

    

#     start_index = random.randint(0, len(text) - maxlen - 1)

    tweet = np.random.choice(text) # select random tweet

    start_index = 0



    for diversity in [0.2, 0.5, 1.0, 1.2]:

        print('----- diversity:', diversity)



        generated = ''

        sentence = tweet[start_index: start_index + maxlen]

        generated += sentence

        print('----- Generating with seed: "' + sentence + '"')

        sys.stdout.write(generated)



        for i in range(120):

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

epochs = 5



print_callback = LambdaCallback(on_epoch_end=on_epoch_end)



model.fit(x, y,

          batch_size=128,

          epochs=epochs,

          callbacks=[print_callback])

print('Build model...')

model2 = Sequential()

model2.add(LSTM(128, input_shape=(maxlen, len(chars)),return_sequences=True))

model2.add(Dropout(0.2))

model2.add(LSTM(128))

model2.add(Dropout(0.2))

model2.add(Dense(len(chars), activation='softmax'))



# optimizer = RMSprop(lr=0.01)

optimizer = Adam()

model2.compile(loss='categorical_crossentropy', optimizer=optimizer)

epochs = 60



print_callback = LambdaCallback(on_epoch_end=on_epoch_end)



model2.fit(x, y,

          batch_size=128,

          epochs=epochs,

          callbacks=[print_callback])

def generate_w_seed(sentence,diversity):

    sentence = sentence[0:maxlen]

    print(f'seed: {sentence}')

    print(f'diversity: {diversity}')

    generated = ''

    generated += sentence

    

    sys.stdout.write(generated)



    for i in range(120):

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

    return
for s in random.sample(list(text),5):

    for diversity in [0.2, 0.5, 1.0, 1.2]:

        generate_w_seed(s,diversity)

        print()