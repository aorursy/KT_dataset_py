# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import string

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import keras.utils as k



# set seeds for reproducability

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(2)

seed(1)
curr_dir = '../input/'

all_headlines = []



for filename in os.listdir(curr_dir):

    if "Articles" in filename:

        article_df = pd.read_csv(curr_dir + filename)

        all_headlines.extend(list(article_df.headline.values))  #append would have made a list inside a list therfore .extend is used

        



all_headlines = [h for h in all_headlines if h!= "Unknown"]
def clean_text(txt):

    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    return txt



corpus = [clean_text(x) for x in all_headlines]
corpus[1:]
tokenizer = Tokenizer()
import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
def get_seq_tokens(corpus):

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    

    #converting data into sequence of tokens

    input_sequences = []

    for line in corpus:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):

            n_gram_sequence = token_list[:i+1]

            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words
print(total_words)
inp_sequences, total_words = get_seq_tokens(corpus)

inp_sequences[:10]
def generate_padded_sequences(input_sequences):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = k.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len



predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
def model(max_sequence_len, total_words):

    input_len = max_sequence_len - 1

    model = Sequential()

    # Add Input Embedding Layer

    model.add(Embedding(total_words, 10, input_length=input_len))

    

    # Add Hidden Layer 1 - LSTM Layer

    model.add(LSTM(100))

    model.add(Dropout(0.1))

    

    # Add Output Layer

    model.add(Dense(total_words, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam')

    

    return model



model = model(max_sequence_len, total_words)

model.summary()
model.fit(predictors, label, epochs=100, verbose=5)
def generate_words(input_word, next_words, model,max_sequence_lennce_lenequence_len):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([input_word])[0]

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=0)

        

        output  = ""

        for word,index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        input_word += " "+output_word

    return input_word.title()
print (generate_words("united states", 5, model, max_sequence_len))

print (generate_words("Engineering", 4, model, max_sequence_len))
