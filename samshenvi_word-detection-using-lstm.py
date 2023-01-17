import numpy as np

import pandas as pd

import os# keras module for building LSTM 

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import keras.utils as ku



import pandas as pd

import numpy as np

import string, os
directory='../input/'

for files in os.listdir('../input/'):

    lines = [line.rstrip('\n') for line in open(directory+files)]

print(lines[1:10])
def clean_text(txt):

    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt 

corpus = [clean_text(x) for x in lines]
tokenizer=Tokenizer(char_level=True)

def get_sequence_of_tokens(corpus):

    ## tokenization

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    for line in corpus:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):

            n_gram_sequence = token_list[:i]

            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
def generate_padded_sequences(input_sequences):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len



predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
def create_model(max_sequence_len, total_words):

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



model = create_model(max_sequence_len, total_words)

model.summary()
model.fit(predictors, label, epochs=50, verbose=3)
def generate_text(seed_text, next_words, model, max_sequence_len):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=2)

        

        output_word = ""

        for word,index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        seed_text += output_word

    return seed_text.title()
inputdata=input("Enter the text:")

print (generate_text(inputdata, 8, model, max_sequence_len))
