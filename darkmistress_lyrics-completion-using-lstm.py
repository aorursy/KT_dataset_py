from numpy import array

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding



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
#Load dataset

filename = '../input/lyricsnow1/lyrics_now1.txt'

file = open (filename, mode = 'r')

data = file.read()







# integer encode sequences of words

tokenizer = Tokenizer()

tokenizer.fit_on_texts([data])

encoded = tokenizer.texts_to_sequences([data])[0]

# retrieve vocabulary size

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

# encode 2 words -> 1 word

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

sequences = array(sequences)

X, y = sequences[:,:-1],sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)

# define model

model = Sequential()

model.add(Embedding(vocab_size, 10, input_length=max_length-1))

model.add(LSTM(50))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

# compile network

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network

model.fit(X, y, epochs=500, verbose=2)

print(generate_seq(model, tokenizer, max_length-1, 'We found', 5))

print(generate_seq(model, tokenizer, max_length-1, 'thin line between', 6))