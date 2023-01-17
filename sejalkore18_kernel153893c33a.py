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
def read_file(filepath):

    

    with open(filepath) as f:

        str_text = f.read()

    

    return str_text

read_file('/kaggle/input/houn.txt')
import spacy

nlp = spacy.load('en',disable=['parser', 'tagger','ner'])



nlp.max_length = 355408

def separate_punc(doc_text):

    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n\n     \n          \n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n \n     \n\n     ']
d = read_file('/kaggle/input/houn.txt')

tokens = separate_punc(d)
# organize into sequences of tokens

train_len = 50+1 # 50 training words , then one target word



# Empty list of sequences

text_sequences = []



for i in range(train_len, len(tokens)):

    

    # Grab train_len# amount of characters

    seq = tokens[i-train_len:i]

    

    # Add to list of sequences

    text_sequences.append(seq)
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences)
import numpy as np
sequences = np.array(sequences)
sequences
vocabulary_size = len(tokenizer.word_counts)
import keras

from keras.models import Sequential

from keras.layers import Dense,LSTM,Embedding
def create_model(vocabulary_size, seq_len):

    model = Sequential()

    model.add(Embedding(vocabulary_size, 50, input_length=seq_len))

    model.add(LSTM(150, return_sequences=True))

    model.add(LSTM(150))

    model.add(Dense(150, activation='relu'))



    model.add(Dense(vocabulary_size, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   

    model.summary()

    

    return model
from keras.utils import to_categorical
X = sequences[:,:-1]

y = sequences[:,-1]
y = to_categorical(y, num_classes=vocabulary_size+1)

seq_len = X.shape[1]
seq_len
model = create_model(vocabulary_size+1, seq_len)
from pickle import dump,load
model.fit(X, y, batch_size=128, epochs=300,verbose=1)
# save the model to file

model.save('epochBIG.h5')

# save the tokenizer

dump(tokenizer, open('epochBIG', 'wb'))