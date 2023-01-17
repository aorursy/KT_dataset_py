# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing the lib keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 

import random
import pandas as pd
import numpy as np
import string, os
# read txt data
with open('/kaggle/input/annatxt/anna.txt','r') as f:
    text = f.read()
#mapping of the text and uisng character wise LSTM


nat_vocab = sorted(set(text))
word2integer = {i: t for i,t in enumerate(nat_vocab)}
int2word = {t: i for i,t in enumerate(nat_vocab)}
######################### this step is to made mapping or semi overlap

seq = 43 #n.of character 
step_shift = 3 #shifting every seq by 3 chracter
## made a list of sentenses
sentences= []

# made a list of next_labeld word[target_value]
nex_word = []

# filling the sentences list by values
for i in range(0 , len(text)- seq,step_shift):
    sentences.append(text[i:i+seq])
    nex_word.append(text[i+seq])

print('Number of sequences:', len(sentences), "\n")

## convert the type of the data to tensor array
# create two tensors one for labels and other for feature
x = np.zeros((len(sentences) ,seq , len(nat_vocab)),dtype =np.bool)
y = np.zeros((len(sentences),len(nat_vocab)),dtype= np.bool)

## fill the x and y with values of character based on sentences


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, int2word[char]] = 1
    y[i, int2word[nex_word[i]]] = 1

# buiidong the model
mod = Sequential()
mod.add(LSTM(200 , input_shape =(seq , len(nat_vocab))))
mod.add(Dense(83,activation='softmax'))
mod.compile(loss = 'categorical_crossentropy',optimizer='rmsprop') 
mod.fit(x,y,batch_size=256,epochs=14)
#define the sample and temp value <<i can play with prop of the genrated text
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - seq - 1)
    generated = ''
    sentence = text[start_index: start_index + seq]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, seq, len(nat_vocab)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, int2word[char]] = 1

        predictions = mod.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = word2integer[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated
print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))
