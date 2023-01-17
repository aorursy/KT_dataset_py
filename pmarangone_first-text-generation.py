import os

print(os.listdir("../input"))

# keras module for building LSTM 

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import keras.utils as ku 



# set seeds for reproducability

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(2)

seed(1)



import pandas as pd

import numpy as np

import string

curr_dir = '../input/'

all_headlines = []

for filename in os.listdir(curr_dir):

    if 'Articles' in filename:

        article_df = pd.read_csv(curr_dir + filename)

        all_headlines.extend(list(article_df.headline.values))

        break



all_headlines = [h for h in all_headlines if h != "Unknown"]

print(len(all_headlines))



#all headlines before taking off punctuation and lower all

# How do the text interpret N.F.L as nfl? How it can be a something useful?

print(all_headlines[:10])
def clean_text(txt):

    #with this list, take each word takeof punctuation put in lower case and join again, with space ("".join)

    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt



#take one list each time of all_headlines

corpus = [clean_text(x) for x in all_headlines]
#All headlines transformed

print(corpus[:10])

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

    return input_sequences, total_words, tokenizer.word_index



inp_sequences, total_words, ID = get_sequence_of_tokens(corpus)

print(inp_sequences[:10])

print(total_words)

print(type(ID))

# # make the numbers the enters of the dict

reverse_word_dict = { value:key for key, value in ID.items() }

print(' '.join(reverse_word_dict[id] for id in inp_sequences[9]))

# print(reverse_word_dict[])
def generate_padded_sequences(input_sequences):

    #input seq é uma lista com varios valores correspondetes a palavras no corpus inicial. 

    #so, max esta tomando o todas os itens de input computando sua extensão e colocando numa outra lista. ent, 

    #MAX está incubido de, nessa lista com todos valores, encontrar a lista com maior extensão

    max_sequence_len = max([len(x) for x in input_sequences])    

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))



    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len



predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
# take predictors and apply reverse, 

# why have so many 000 after padded?



# print(predictors[11])

# t = [99, 269,  371, 1115,  582,   52,    7,    2,  372,   10, 100]

# for i in t:

#     print(reverse_word_dict[i])



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
model.fit(predictors, label, epochs=100, verbose=5)

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









#esse modelo foi criado a partir de manchestes do newyorktimes, ou seja, 

#esperar que seja capaz de gerar algo além do escopo de uma manchete (seja em extensão ou formalidade)

#é trivial



#FastText was build on top of Wikipedia, sooooo

#GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset

#of 8 million web pages. 



print (generate_text("united states", 5, model, max_sequence_len))

print (generate_text("preident trump", 4, model, max_sequence_len))

print (generate_text("donald trump", 4, model, max_sequence_len))

print (generate_text("india and china", 4, model, max_sequence_len))

print (generate_text("new york", 4, model, max_sequence_len))

print (generate_text("science and technology", 5, model, max_sequence_len))