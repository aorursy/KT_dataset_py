import warnings

warnings.filterwarnings('ignore')

#ignoring tensorflow warnings



import string

import re

import os

import numpy as np

import pandas as pd 

import nltk

#keras related imports

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_path = os.path.join(dirname, filename)
ps = nltk.PorterStemmer()
def clean_text(text):

    text = text.replace('--', ' ')

    tokens = text.split()

    table = str.maketrans('', '', string.punctuation)

    tokens = [w.translate(table) for w in tokens]

    tokens = [word for word in tokens if word.isalpha()]

    tokens = [word.lower() for word in tokens if word not in nltk.corpus.stopwords.words('english')]

    return tokens
def gen_data(rows):

    lyrics_data = pd.read_csv(data_path)

    print("No of rows in data are %s",lyrics_data.__len__())

    lyrics_data = lyrics_data[pd.notnull(lyrics_data['lyrics'])]

    if rows<lyrics_data.__len__(): 

        lyrics_data_sample = lyrics_data[0:rows]

        lyrics_data_sample['lyrics'] = lyrics_data_sample['lyrics'].apply(lambda x: clean_text(x))

    else:

        print("Rows exceeded")

    return lyrics_data_sample
data = gen_data(10000)
tokens = list()

for row in data['lyrics']:

    tokens += row
length = 51 ## no of words in each sequence

lines = list()

for i in range(0,len(tokens)-len(tokens)%length,length):

    seq = tokens[i:i+length]

    line = ' '.join(seq)

    lines.append(line)

print('Total Sequences: %d' % len(lines))
lines[0]
tokenizer = Tokenizer()

tokenizer.fit_on_texts(lines)

sequences = tokenizer.texts_to_sequences(lines)
sequences = np.array(sequences)

vocab_size = len(tokenizer.word_index) + 1
vocab_size
X, y = sequences[:,:-1], sequences[:,-1]

seq_length = X.shape[1]

y = to_categorical(y, num_classes=vocab_size)
model = Sequential()

model.add(Embedding(vocab_size, 50, input_length=seq_length))

model.add(LSTM(100, return_sequences=True))

#model.add(LSTM(100, return_sequences=True))

model.add(LSTM(100))

model.add(Dense(100, activation='relu'))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model).create(prog='dot', format='svg'))
# compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model

model.fit(X, y, batch_size=128, epochs=50)
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):

    result = list()

    in_text = seed_text

    # generate a fixed number of words

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
from random import randint



generated = generate_seq(model, tokenizer, 50, lines[randint(0,len(lines))], 50)

print(generated)