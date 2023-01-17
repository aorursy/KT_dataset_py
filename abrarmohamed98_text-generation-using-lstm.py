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
from numpy import array

from pickle import dump

from keras.preprocessing.text import Tokenizer

from keras.utils.vis_utils import plot_model

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

import string

import re
def save_doc(lines, filename):

    data = '\n'.join(lines)

    file = open(filename, 'w')

    file.write(data)

    file.close()
def load_doc(filename):

    file = open(filename, 'r')

    # read all text

    text = file.read()

    # close the file

    file.close()

    return text

    
in_filename='../input/ali.txt'
doc=load_doc(in_filename)
print(doc[0:100])
def clean_doc(doc):

    doc = doc.replace('--', ' ')

    tokens = doc.split()

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    # remove punctuation from each word

    tokens = [re_punc.sub('', w) for w in tokens]

    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]

    # make lower case

    tokens = [word.lower() for word in tokens]

    return tokens
tokens=clean_doc(doc)

tokens[0:50]

tk=len(tokens)

tk
length= 50 + 1

sequence=[]

for i in range(length,tk):

    seq=tokens[i-length:i]

    line=' '.join(seq)

    sequence.append(line)

    
sequence
out_filename = 'ali1.txt'

save_doc(sequence, out_filename)
len(sequence)
tokenizer=Tokenizer()

tokenizer.fit_on_texts(sequence)
sequence1 = tokenizer.texts_to_sequences(sequence)
sequence1
vocab_size=len(tokenizer.word_index) + 1

print(vocab_size)

sequence1=np.array(sequence1)
sequence1
x,y=sequence1[:,:-1],sequence1[:,-1]
y = to_categorical(y, num_classes=vocab_size)
print("shape of training datasets(x,y)=",x.shape,y.shape)
def define_model(vocab_size, seq_length):

    model = Sequential()

    model.add(Embedding(vocab_size, 50, input_length=seq_length))

    model.add(LSTM(100, return_sequences=True))

    model.add(LSTM(100))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
x.shape[1]
model=define_model(vocab_size,x.shape[1])
model.fit(x,y,batch_size=128,epochs=170)
sequence
from keras.models import load_model
model.save('model.h5')
model = load_model('model.h5')
sequence
from random import randint

from pickle import load

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences

# load doc into memory

def load_doc(filename):

    # open the file as read only

    file = open(filename, 'r')

    # read all text

    text = file.read()

    # close the file

    file.close()

    return text

# generate a sequence from a language model

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

# load cleaned text sequences

in_filename = 'ali1.txt'

doc = load_doc(in_filename)

lines = doc.split('\n')

seq_length = len(lines[0].split()) - 1

# select a seed text

seed_text = 'Ali Baba told his brother the secret of the cave. Cassim rose early next morning, and set out with ten mules loaded with great chests.'

print(seed_text + '\n')

# generate new text

generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)

print(generated)
