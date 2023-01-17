import re

import keras as k

import numpy as np

import pandas as pd

from pickle import dump

from random import randint

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding

from keras.preprocessing.sequence import pad_sequences
riddles = pd.read_csv('/kaggle/input/output.csv')

riddles['quiz'] = riddles['quiz'].apply((lambda x: re.sub('[^а-яА-я0-9\s]','',x.lower())))

print(riddles['quiz'])
tokenizer = k.preprocessing.text.Tokenizer(split=' ')

tokenizer.fit_on_texts(riddles['quiz'].values)

sequences = tokenizer.texts_to_sequences(riddles['quiz'].values)

sequences = k.preprocessing.sequence.pad_sequences(sequences)

sequence_dict = tokenizer.word_index;
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)
X, y = sequences[:,:-1], sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

riddle_num = X.shape[0]

print(seq_length, riddle_num)
model = Sequential()

model.add(Embedding(vocab_size, 50, input_length=seq_length))

model.add(LSTM(100, return_sequences=True))

model.add(LSTM(100))

model.add(Dense(100, activation='relu'))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, batch_size=128, epochs=16)
model.save('model.h5')

dump(tokenizer, open('tokenizer.pkl', 'wb'))
seed_text = riddles['quiz'][randint(0, riddle_num)]

print(seed_text)
encoded = tokenizer.texts_to_sequences([seed_text])[0]

encoded = np.pad(encoded, (seq_length - len(encoded), 0), 'constant')

encoded = np.reshape(encoded, (1, seq_length))

print(encoded)
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):

    result = list()

    in_text = seed_text

    for _ in range(n_words):

        # encode the text as integer

        encoded = tokenizer.texts_to_sequences([in_text])[0]

        # truncate sequences to a fixed length

        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')

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

        result.append(out_word)

    return ' '.join(result)
generated = generate_seq(model, tokenizer, seq_length, seed_text, randint(6, 18))

print(generated)