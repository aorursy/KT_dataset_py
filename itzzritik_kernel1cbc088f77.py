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
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense, Dropout,Bidirectional

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

import keras.utils as ku

import string
import pandas as pd

import numpy as np
curr_dir = '../input/'

reviews = []

for file in os.listdir(curr_dir):

    review9_df = pd.read_csv(curr_dir + file)

    reviews.extend(list(review9_df.body.values))



len(reviews)
def clean_text(txt):

    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt 



corpus = [clean_text(x) for x in reviews]

corpus[:10]
tokenizer = Tokenizer()



def get_sequence_of_tokens(corpus):

    ## tokenization

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    

    ## convert data to sequence of tokens 

    input_sequences = []

    for line in corpus:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):

            n_gram_sequence = token_list[:i+1]

            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words



inp_sequences, total_words = get_sequence_of_tokens(corpus)

inp_sequences[:10]
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

    model.add(Embedding(total_words, 10, input_length=input_len))

    model.add(Bidirectional(LSTM(units=256)))

    model.add(Dropout(0.4))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam') 

    return model





model = create_model(max_sequence_len, total_words)
#model.load_weights("model_weights.h5")
model.fit(predictors, label, epochs=100,batch_size=100)
def generate_text(seed_text, next_words, model, max_sequence_len):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=0)

        

        output_word = ""

        for word,index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        seed_text += " "+output_word

    return seed_text.title()
model.save_weights("model_weights.h5")
print (generate_text("this phone", 15, model, max_sequence_len))