# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Flatten, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import Model
from keras.models import model_from_json
import keras.utils as ku 
import keras

# set seeds for reproducability
from tensorflow import set_random_seed
import tensorflow as tf
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 
import collections
import random

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# loading dataset
dire = '../input/nyt-comments/'
headlines = []
FULLDATA = True
for filename in os.listdir(dire):
    if 'Articles' in filename:
        print('processing', filename, '...')
        article_df = pd.read_csv(dire + filename)
        headlines.extend(list(article_df.headline.values))
    if not FULLDATA:
        break
print('DONE')
headlines[-10:]
embeddings_index = dict()
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = vector
embedding_size = len(embeddings_index['a'])
embedding_size
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 
def mapToFlattenList(token_list):
    flatten_list = []
    for token in token_list:
        if token in embeddings_index:
            flatten_list += list(embeddings_index[token])
        else:
            flatten_list += [0] * embedding_size
    return flatten_list
max_n_gram = 10
tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = text_to_word_sequence(line)
        for i in range(1, len(token_list)):
            n_gram_sequence = mapToFlattenList(token_list[max(i-max_n_gram,0):i])
            input_sequences.append(n_gram_sequence+tokenizer.texts_to_sequences([token_list[i]])[0])
    return input_sequences, total_words
corpus = [clean_text(h) for h in headlines]
corpus[:5]
inp_sequences, total_words = get_sequence_of_tokens(corpus)
# padding sequences and attach labels
def generate_padded_sequences(input_sequences):
    n = len(input_sequences)
    # max_sequence_len = maximum length of n gram in input sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre',dtype='float'))
    predictors, label = input_sequences[:,:-1].reshape((n, -1 ,embedding_size)),input_sequences[:,-1:]
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
predictors.shape, label.shape
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    # n * 17 inputs
    # n * 10 outputs
    #     model.add(Embedding(total_words, 10, input_length=input_len))
    #     model.add(Embedding(total_words, 100, weights=[embedding_matrix], input_length=input_len))
    model.add(Bidirectional(LSTM(units=256,
                activation='relu'), input_shape=predictors[0].shape))
    
    model.add(Dropout(0.4))
    
    model.add(Dense(total_words, activation='softmax'))
    
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam') 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adagrad') 
    return model

model = create_model(max_sequence_len, total_words)
model.summary()
model.fit(predictors, label, epochs=100, verbose=2)
# next words: number of words to be generated 
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        # token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = text_to_word_sequence(seed_text)
        n_gram_sequence = mapToFlattenList(token_list[max(len(token_list)-max_n_gram,0):])
        input_sequences = np.array(pad_sequences([n_gram_sequence], maxlen=max_sequence_len-1, padding='pre',dtype='float'))
        predictor = input_sequences.reshape((1, -1 ,embedding_size))
        predicted = model.predict_classes(predictor, verbose=0)
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
generate_text("California", 10, model, max_sequence_len)
generate_text("new york", 10, model, max_sequence_len)
generate_text("united states", 10, model, max_sequence_len)
generate_text("preident trump", 10, model, max_sequence_len)
generate_text("donald trump", 10, model, max_sequence_len)
generate_text("india and china", 10, model, max_sequence_len)
generate_text("science and technology", 10, model, max_sequence_len)
generate_text("How", 10, model, max_sequence_len)
generate_text("Where", 10, model, max_sequence_len)
generate_text("Why", 10, model, max_sequence_len)
generate_text("When", 10, model, max_sequence_len)
