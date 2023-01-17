from __future__ import print_function

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.layers import LSTM

from keras.optimizers import RMSprop, Adam

from keras.utils.data_utils import get_file

import numpy as np

import pandas as pd

import random

import sys



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read the entire file containing song lyrics

path = "../input/songlyrics/songdata.csv"

df = pd.read_csv(path)

df.head()
text = df['text'].str.cat(sep='\n').lower()



# Output the length of the corpus

print('corpus length:', len(text))



# Create a sorted list of the characters

chars = sorted(list(set(text)))

print('total chars:', len(chars))



# Corpus is going to take way too long for me to train, so lets make it shorter... 

# Pretty sure this is just the artists under "A"

text = text[:1000000]

print('truncated corpus length:', len(text))
# Create a dictionary where given a character, you can look up the index and vice versa

char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))



# cut the text in semi-redundant sequences of maxlen characters

maxlen = 40 # The window size

step = 3 # The steps between the windows

sentences = []

next_chars = []



# Step through the text via 3 characters at a time, taking a sequence of 40 bytes at a time. 

# There will be lots ofo overlap

for i in range(0, len(text) - maxlen, step):

    sentences.append(text[i: i + maxlen]) # range from current index i for max length characters 

    next_chars.append(text[i + maxlen]) # the next character after that 

sentences = np.array(sentences)

next_chars = np.array(next_chars)

print('Number of sequences:', len(sentences))


def getdata(sentences, next_chars):

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    length = len(sentences)

    index = 0

    for i in range(len(sentences)):

        sentence = sentences[i]

        for t, char in enumerate(sentence):

            X[i, t, char_indices[char]] = 1

        y[i, char_indices[next_chars[i]]] = 1

    return X, y



def generator(sentences, next_chars, batch_size):

    X = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)

    y = np.zeros((batch_size, len(chars)), dtype=np.bool)

    length = len(sentences)

    index = 0

    while True:

        if index + batch_size >= length:

            index = 0

        X.fill(0)

        y.fill(0)

        for i in range(batch_size):

            sentence = sentences[index]

            for t, char in enumerate(sentence):

                X[i, t, char_indices[char]] = 1

            y[i, char_indices[next_chars[i]]] = 1

            index = index + 1

        yield X, y
# build the model: a single LSTM

print('Build model...')

model = Sequential()

model.add(LSTM(128, input_shape=(maxlen, len(chars))))

model.add(Dense(len(chars)))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



print("Compiling model complete...")

def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)
trained = True



if not trained:

    # Get data

    X, y = getdata(sentences, next_chars)



    # The training

    print('Training...')

    batch_size = 128



    # Use the below command if you want to use the generator

    # history = model.fit_generator(generator(sentences, next_chars, batch_size),steps_per_epoch=12800, epochs=10)



    # Use this if they all fit into memory

    history = model.fit(X, y,batch_size=128, epochs=35)





    # Save the model

    model.save('songgenerator.h5')

    

else:

    model.load_weights('../input/songgeneratormodel/songgenerator.h5')


# Given a seed, lets check out what our model predicts

sentence = 'the duck walked up to the lemonade stand'

print(len(sentence))

x = np.zeros((1, maxlen, len(chars)))

for t, char in enumerate(sentence):

    x[0, t, char_indices[char]] = 1.

    

print(model.predict(x, verbose=0)[0])

print(sum(model.predict(x, verbose=0)[0]))
# Variance is used by the sample function, that will randomly select

# the next most probable character from the softmax outpu 

variance = 0.2

print('Variance: ', variance)



generated = ''

original = sentence

window = sentence

# Predict the next 400 characters based on the seed

for i in range(400):

    x = np.zeros((1, maxlen, len(chars)))

    for t, char in enumerate(window):

        x[0, t, char_indices[char]] = 1.



    preds = model.predict(x, verbose=0)[0]

    next_index = sample(preds, variance)

    next_char = indices_char[next_index]



    generated += next_char

    window = window[1:] + next_char



print(original + generated)
