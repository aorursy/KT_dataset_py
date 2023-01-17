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
# importing libraries

# keras module for building LSTM



from keras.models import Sequential

from keras.layers import Embedding, Dense, Dropout, LSTM

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import keras.utils as ku

from keras.callbacks import EarlyStopping



# set seeds for reproducability



from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(2)

seed(1)



import string

import os

import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter(action='ignore',  category=FutureWarning)
currr_dir = "../input/"

all_headlines = []



x=0



for filename in os.listdir(currr_dir):

    if 'Articles' in filename:

        

        if x==0:

            print(filename)

            

        article_df = pd.read_csv(currr_dir+filename)

        

        if x==0:

            print(article_df.shape)

            print(article_df.columns)

            print(article_df.head(5))

            print(article_df.tail(5))

            

        all_headlines.extend(list(article_df.headline.values))

        

        if x==0:

            print(article_df.headline)

            print(article_df.headline.values)

            

        x=1

        

        break



all_headlines = [ h for h in all_headlines if h != 'Unknown' ]

print(len(all_headlines))

print(all_headlines[:5])
def clean_text(txt):

    txt = "".join(w for w in txt if w not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii","ignore")

    return txt
print(clean_text("Questions for: ‘Colleges Discover the Rural St.."))
print(string.punctuation)
corpus = [clean_text(x) for x in all_headlines]

print(corpus[:10])
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):

    q=0

    # tokenization

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    

    if q==0:

        print(total_words)

    

    # convert data into sequence of tokens

    input_sequences = []

    for line in corpus:

        

        if q==0:

            print(line)

            

        token_list = tokenizer.texts_to_sequences([line])[0]

        

        if q==0:

            print(token_list)

            print(len(token_list))

            

        for i in range(1, len(token_list)):

            n_gram_sequence = token_list[:i+1]

            

            if q==0:

                print(n_gram_sequence)

                

            input_sequences.append(n_gram_sequence)

            

            if q==0:

                print(input_sequences)

                

            q=1

    

    return input_sequences, total_words
input_sequence, total_words = get_sequence_of_tokens(corpus)

print(total_words)
(input_sequence[:10])
def generate_padded_sequences(input_sequence):

    max_sequence_len = max([len(x) for x in input_sequence])

    input_sequences = np.array(pad_sequences(input_sequence, maxlen=max_sequence_len, padding='pre'))

    predictors, labels = input_sequences[:,:-1], input_sequences[:,-1]

    labels = ku.to_categorical(labels, num_classes=total_words)

    

    return predictors, labels, max_sequence_len
predictors, labels, max_seq_len = generate_padded_sequences(input_sequence)
len(predictors)
len(labels)
(max_seq_len)
print(total_words)
def create_model(max_seq_len, total_words):

    input_len = max_seq_len - 1

    

    model = Sequential()

    

    # input: embedding layer

    model.add(Embedding(total_words, 10, input_length=input_len))

    

    # hidden: lstm layer

    model.add(LSTM(100))

    model.add(Dropout(0.1))

    

    # output layer

    model.add(Dense(total_words, activation='softmax'))

    

    # compile the model

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    

    return model
model = create_model(max_seq_len, total_words)

model.summary()
# fitting (training) the model by passing predictors and labels as training data



model.fit(predictors, labels, epochs=100, verbose=5)
def generate_text(seed_text, next_words, model, max_seq_len):

    w=0

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        

        if w==0:

            print(token_list)

            

        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        

        if w==0:

            print(token_list)

        

        predicted = model.predict_classes(token_list, verbose=0)

        

        if w==0:

            print(predicted)

        

        output_word = ''

        

        for word,index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

                

        seed_text = seed_text + " " + output_word

        

        w=1

        

    return seed_text.title()
print(generate_text("united states", 5, model, max_seq_len))
print (generate_text("united states", 5, model, max_seq_len))

print (generate_text("preident trump", 4, model, max_seq_len))

print (generate_text("donald trump", 4, model, max_seq_len))

print (generate_text("india and china", 4, model, max_seq_len))

print (generate_text("new york", 4, model, max_seq_len))

print (generate_text("science and technology", 5, model, max_seq_len))