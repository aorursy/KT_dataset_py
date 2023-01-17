import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import layers

from keras.models import Sequential

import keras.utils as ku

from keras.callbacks import EarlyStopping
# Loading the dataset

data = pd.read_json('/kaggle/input/quotes-dataset/quotes.json')

print(data.shape)

data.head()
# Dropping duplicates and creating a list containing all the quotes

quotes = data['Quote'].drop_duplicates()

print(f"Total Unique Quotes: {quotes.shape}")



# Considering only top 3000 quotes

quotes_filt = quotes.sample(3000)

print(f"Filtered Quotes: {quotes_filt.shape}")

all_quotes = list(quotes_filt)

all_quotes[:2]
# Tokeinization

tokenizer = Tokenizer()



# Function to create the sequences

def generate_sequences(corpus):

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    print(f"Total unique words in the text corpus: {total_words}")

    input_sequences = []

    for line in corpus:

        seq = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(seq)):

            ngram_seq = seq[:i+1]

            input_sequences.append(ngram_seq)

            

    return input_sequences, total_words



# Generating sequences

input_sequences, total_words = generate_sequences(all_quotes)

input_sequences[:5]
# Generating predictors and labels from the padded sequences

def generate_input_sequence(input_sequences):

    maxlen = max([len(x) for x in input_sequences])

    input_sequences = pad_sequences(input_sequences, maxlen=maxlen)

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, maxlen



predictors, label, maxlen = generate_input_sequence(input_sequences)

predictors[:1], label[:1]
# Building the model

embedding_dim = 64



def create_model(maxlen, embedding_dim, total_words):

    model = Sequential()

    model.add(layers.Embedding(total_words, embedding_dim, input_length = maxlen))

    model.add(layers.LSTM(128, dropout=0.2))

    model.add(layers.Dense(total_words, activation='softmax'))

    

    # compiling the model

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model



model = create_model(maxlen, embedding_dim, total_words)

model.summary()
predictors.shape , label.shape, maxlen
# Training the model

# model.fit(predictors, label, epochs=50, batch_size=64)
# Save the model for later use

# model.save("Quotes_generator.h5")
# Loading the model

from keras.models import load_model



Quotes_gen = load_model("../input/quote-generator-trained-model/Quotes_generator.h5")
Quotes_gen.summary()
# Text generating function

def generate_quote(seed_text, num_words, model, maxlen):

    

    for _ in range(num_words):

        tokens = tokenizer.texts_to_sequences([seed_text])[0]

        tokens = pad_sequences([tokens], maxlen=maxlen, padding='pre')

        

        predicted = model.predict_classes(tokens)

        

        output_word = ''

        

        for word, index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        seed_text = seed_text + " " + output_word

    

    return seed_text
# Let's try to generate some quotes

print(generate_quote("Passion", num_words = 10, model= Quotes_gen, maxlen=maxlen))
print(generate_quote("Love", num_words = 20, model= Quotes_gen, maxlen=maxlen))
print(generate_quote("legend", num_words = 15, model= Quotes_gen, maxlen=maxlen))
print(generate_quote("consistency matters", num_words = 15, model= Quotes_gen, maxlen=maxlen))
print(generate_quote("Follow your passion", num_words = 20, model= Quotes_gen, maxlen=maxlen))