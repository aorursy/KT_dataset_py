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
file=open('../input/wonderland.txt',encoding='utf-8') 
raw_text=file.read()
raw_text #it reads file

raw_text=raw_text.lower()
set(raw_text)  #making set

list(set(raw_text))

chars=sorted(list(set(raw_text)))

print(chars)
bad_chars=['#','*','_','@','\ufeff']

for i in range(len(bad_chars)):

  raw_text=raw_text.replace(bad_chars[i],"")

chars=sorted(list(set(raw_text)))

print(chars)
text_len=len(raw_text)

char_len=len(chars)

print(text_len)

print(char_len)
import numpy as np

import keras

from keras.models import Sequential

from keras.layers import Dense, CuDNNLSTM, Dropout, Activation

from keras.optimizers import RMSprop, Adam

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils





seq_len=100
char_to_int=dict((c,i) for i,c in enumerate(chars))

char_to_int #convert into dictonary with numeric_label







input_str=[]

output_str=[]



for i in range(len(raw_text)-seq_len):

  x_test=raw_text[i:i+seq_len]

  x=[char_to_int[char] for char in x_test]  #take numeric data

  input_str.append(x)

  y=raw_text[i+seq_len]

  output_str.append(char_to_int[y])

vocab = char_len

vocab
def buildmodel(vocab):

    model = Sequential()

    model.add(CuDNNLSTM(256, input_shape = (seq_len, 1), return_sequences = True))

    model.add(Dropout(0.2))

    model.add(CuDNNLSTM(256))

    model.add(Dropout(0.2))

    model.add(Dense(vocab, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    return model
length = len(input_str)

input_str = np.array(input_str)

input_str = np.reshape(input_str, (input_str.shape[0], input_str.shape[1], 1))

print(input_str.shape)

output_str = np.array(output_str)

output_str = np_utils.to_categorical(output_str)

print(output_str.shape)
model = buildmodel(vocab)

#filepath="saved_models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

#callbacks_list = [checkpoint]



history = model.fit(input_str, output_str, epochs = 10, batch_size = 128)
initial_text = ' the sun did not shine, it was too wet to play, so we sat in the house all that cold, cold wet day. '# we sat here we two and we said how we wish we had something to do.'

len(initial_text)
initial_text1 = [char_to_int[c] for c in initial_text]

print(initial_text1)
GENERATED_LENGTH = 1000

test_text = initial_text1

generated_text = []

int_to_char=dict((i,c) for i,c in enumerate(chars))



for i in range(1000):

    X = np.reshape(test_text, (1, seq_len, 1))

    next_character = model.predict(X)

    index = np.argmax(next_character)

    generated_text.append(int_to_char[index])

    test_text.append(index)

    test_text = test_text[1:]
generated_text