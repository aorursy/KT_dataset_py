# source text

data = """ Jack and Jill went up the hill\n

To fetch a pail of water\n

Jack fell down and broke his crown\n

And Jill came tumbling after\n """
from keras.preprocessing.text import Tokenizer

# integer encode text

tokenizer = Tokenizer()

tokenizer.fit_on_texts([data])

encoded = tokenizer.texts_to_sequences([data])[0]
# determine the vocabulary size

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)
# create word -> word sequences

sequences = []

for i in range(1, len(encoded)):

    sequence = encoded[i-1:i+1]   

    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))
# split into X and y elements

import numpy as np

import pandas as pd

sequences = np.array(sequences)

X, y = sequences[:,0],sequences[:,1]
from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

# one hot encode outputs

y = to_categorical(y, num_classes=vocab_size)
# define the model

def define_model(vocab_size):

    model = Sequential()

    model.add(Embedding(vocab_size, 10, input_length=1))

    model.add(LSTM(50))

    model.add(Dense(vocab_size, activation='softmax'))

    # compile network

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model

    model.summary()

    plot_model(model, to_file='model.png', show_shapes=True)

    return model
model = define_model(vocab_size) # build the model
model.fit(X, y, epochs=500, verbose=2) # fit the model
# evaluate

in_text = 'Jack'

print(in_text)

encoded = tokenizer.texts_to_sequences([in_text])[0]

encoded = np.array(encoded)

yhat = model.predict_classes(encoded, verbose=0)

for word, index in tokenizer.word_index.items():

    if index == yhat:

        print(word)
# generate a sequence from the model

def generate_seq(model, tokenizer, seed_text, n_words):

    in_text, result = seed_text, seed_text

    # generate a fixed number of words

    for _ in range(n_words):

        # encode the text as integer

        encoded = tokenizer.texts_to_sequences([in_text])[0]

        encoded = np.array(encoded)

        # predict a word in the vocabulary

        yhat = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word

        out_word = ''

        for word, index in tokenizer.word_index.items():

            if index == yhat:

                out_word = word

                break

    # append to input

        in_text, result = out_word, result + ' ' + out_word

    return result
print(generate_seq(model, tokenizer, 'Jack', 6))
# create line-based sequences

sequences = list()

for line in data.split('\n'):

    encoded = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(encoded)):

        sequence = encoded[:i+1]

        sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))
from keras.preprocessing.sequence import pad_sequences

#pad input sequences

max_length = max([len(seq) for seq in sequences])

sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

print('Max Sequence Length: %d' % max_length)
# split into input and output elements

sequences = np.array(sequences)

X, y = sequences[:,:-1],sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)
# define the model

def define_model(vocab_size, max_length):

    model = Sequential()

    model.add(Embedding(vocab_size, 10, input_length=max_length-1))

    model.add(LSTM(50))

    model.add(Dense(vocab_size, activation='softmax'))

    # compile network

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model

    model.summary()

    plot_model(model, to_file='model.png', show_shapes=True)

    return model
# generate a sequence from a language model

def generate_seq(model, tokenizer, max_length, seed_text, n_words):

    in_text = seed_text

    # generate a fixed number of words

    for _ in range(n_words):

        # encode the text as integer

        encoded = tokenizer.texts_to_sequences([in_text])[0]

        # pre-pad sequences to a fixed length

        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')

        # predict probabilities for each word

        yhat = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word

        out_word = ''

        for word, index in tokenizer.word_index.items():

            if index == yhat:

                out_word = word

                break

        # append to input

        in_text += ' ' + out_word

    return in_text
# define model

model = define_model(vocab_size, max_length)

# fit network

model.fit(X, y, epochs=500, verbose=2)
# evaluate model

print(generate_seq(model, tokenizer, max_length-1, 'Jack', 4))

print(generate_seq(model, tokenizer, max_length-1, 'Jill', 4))
# generate a sequence from a language model

def generate_seq(model, tokenizer, max_length, seed_text, n_words):

    in_text = seed_text

    # generate a fixed number of words

    for _ in range(n_words):

        # encode the text as integer

        encoded = tokenizer.texts_to_sequences([in_text])[0]

        # pre-pad sequences to a fixed length

        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')

        # predict probabilities for each word

        yhat = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word

        out_word = ''

        for word, index in tokenizer.word_index.items():

            if index == yhat:

                out_word = word

                break

                # append to input

        in_text += ' ' + out_word

    return in_text
# tokenize the data

tokenizer = Tokenizer()

tokenizer.fit_on_texts([data])

encoded = tokenizer.texts_to_sequences([data])[0]
sequences = list()

for i in range(2, len(encoded)):

    sequence = encoded[i-2:i+1]

    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))
# pad sequences

max_length = max([len(seq) for seq in sequences])

sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

print('Max Sequence Length: %d' % max_length)
# split into input and output elements

sequences = np.array(sequences)

X, y = sequences[:,:-1],sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)
# define model

model = define_model(vocab_size, max_length)

# fit network

model.fit(X, y, epochs=500, verbose=2)
# evaluate model

print(generate_seq(model, tokenizer, max_length-1, 'Jack and', 5))

print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))

print(generate_seq(model, tokenizer, max_length-1, 'fell down', 5))

print(generate_seq(model, tokenizer, max_length-1, 'pail of', 5))