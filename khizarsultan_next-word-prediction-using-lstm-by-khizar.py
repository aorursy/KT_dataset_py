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
file = open('/kaggle/input/poem.txt','r')
text = file.read()
text
file.close()
tokens = text.split()

raw_text = ' '.join(tokens)
raw_text
#     # organize into sequences of characters 

#     length = 10 

#     sequences = list() 

#     for i in range(length, len(raw_text)): 

#         # select sequence of tokens 

#         seq = raw_text[i-length:i+1]



length = 10

sequences = list()

for i in range(length, len(raw_text)):

    # select sequence of tokens

    seq = raw_text[i-length:i+1]

    sequences.append(seq)    
sequences[:100]
# Data has been in converted into sequence

print(f"Total Sequences are {len(sequences)}")
data = '\n'.join(sequences) 

file = open('char_sequences', 'w') 

file.write(data) 

file.close()
chars = sorted(list(set(text)))
mapping = dict((c, i) for i, c in enumerate(chars))
mapping
lines
sequences_encoded = list()

for line in sequences:

    encoded_seq = [mapping[char] for char in line]

    sequences_encoded.append(encoded_seq)

    
#Sequence coding of sequences Strings

sequences_encoded

# vocabulary size 

vocab_size = len(mapping) 

print('Vocabulary Size: %d' % vocab_size)
from numpy import array 

from pickle import dump 

from keras.utils import to_categorical 

from keras.utils.vis_utils import plot_model 

from keras.models import Sequential 

from keras.layers import Dense 

from keras.layers import LSTM
sequences = array(sequences_encoded)
sequences
X,y = sequences[:,:-1],sequences[:,-1]
# vocabulary size 

vocab_size = len(mapping)
vocab_size
# one hot encoding of features

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
X
y = to_categorical(y, num_classes=vocab_size)
y
# define the model 

def define_model(X): 

    model = Sequential() 

    model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2]))) 

    model.add(Dense(vocab_size, activation='softmax')) 

    # compile model 

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # summarize defined model 

    model.summary() 

    plot_model(model, to_file='model.png', show_shapes=True) 

    return model
model = define_model(X) # fit model 

model.fit(X, y, epochs=100, verbose=2)


# save the model to file 

model.save('model.h5') # save the mapping

dump(mapping, open('mapping.pkl', 'wb'))
from pickle import load 

from keras.models import load_model 

from keras.utils import to_categorical 

from keras.preprocessing.sequence import pad_sequences

# generate a sequence of characters with a language model 

def generate_seq(model, mapping, seq_length, seed_text, n_chars): 

    in_text = seed_text # generate a fixed number of characters 

    for _ in range(n_chars): # encode the characters as integers 

        encoded = [mapping[char] for char in in_text] # truncate sequences to a fixed length 

        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre') # one hot encode 

        encoded = to_categorical(encoded, num_classes=len(mapping)) 

        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1]) # predict character 

        yhat = model.predict_classes(encoded, verbose=0) # reverse map integer to character 

        out_char = '' 

        for char, index in mapping.items(): 

            if index == yhat: 

                out_char = char 

                break # append to input 

        in_text += out_char 

    return in_text

# # load the model 

# model = load_model('model.h5') 

# # load the mapping

# mapping = load(open('mapping.pkl', 'rb')) 

# # test start of rhyme 

# print(generate_seq(model, mapping, 10, 'Sing a son', 20)) 

# # test mid-line 

# print(generate_seq(model, mapping, 10, 'king was i', 20)) 

# # test not in original 

# print(generate_seq(model, mapping, 10, 'hello worl', 20))