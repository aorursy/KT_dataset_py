import pandas as pd

import tensorflow as tf

from tensorflow import keras

import numpy as np
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames: 

        if filename == 'True.csv':

            true_news = pd.read_csv(os.path.join(dirname, filename))

        else:

            fake_news = pd.read_csv(os.path.join(dirname, filename))
print(' --- Real News --- ')

print(true_news.head(5))

print(' --- Fake News --- ')

print(fake_news.head(5))
# Let's indicate true news as 1 and fake news as 0

true_news['truth'] = len(true_news) * [1]

fake_news['truth'] = len(fake_news) * [0]

print(true_news.head(3))

print(fake_news.head(3))
# Let's concatenate the two and prepare to shuffle up the data

all_news = pd.concat([true_news, fake_news], ignore_index=True) # ignore index to reset the indices



# Shuffle

all_news = all_news.sample(frac=1).reset_index(drop=True)



print(all_news.head(5))
# Let's only keep title, text, and truth for now

all_news = all_news.drop(['date'], axis=1)

all_news = all_news.drop(['subject'], axis=1)

print(all_news.head(5))
# In this case, let's tokenize and then split

# First, let's find how big the vocabulary is without truncating

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer()

tokenizer.fit_on_texts(all_news['text']) # tokenizing just on text should be fine

word_index = tokenizer.word_index

print(len(word_index))

# 138021 in total

del(tokenizer)
# Great, now let's do two splits on this dataset: one between train/validation and test,

# and another between train and validation

test_breakpoint = int(0.8 * len(all_news))

train_val_data = all_news[:test_breakpoint]

test_data = all_news[test_breakpoint:]



val_breakpoint = int(0.8 * len(train_val_data))

train_data = train_val_data[:val_breakpoint]

val_data = train_val_data[val_breakpoint:]



print(f'Length of training data: {len(train_data)}')

print(f'Length of validation data: {len(val_data)}')

print(f'Length of test data: {len(test_data)}')

print(f'Length of all news: {len(all_news)}')

# Numbers add up!
# Let's tokenize

vocab_size = 50000

embedding_dim = 64

max_length = 150

max_length_title = 50

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>'



# Separate train data

train_text = train_data['text']

train_titles = train_data['title']

train_truth = train_data['truth']



# Separate validation data

val_text = val_data['text']

val_titles = val_data['title']

val_truth = val_data['truth']



# Separate test data

test_text = test_data['text']

test_titles = test_data['title']

test_truth = test_data['truth']



# Get the vocabulary

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(train_text)

word_index = tokenizer.word_index



# Tokenize the train texts

train_text_sequences = tokenizer.texts_to_sequences(train_text)

train_text_padded = np.array(pad_sequences(train_text_sequences, maxlen = max_length, \

                               padding=padding_type, truncating = trunc_type))



# Tokenize the train titles, with lower max length

train_titles_sequences = tokenizer.texts_to_sequences(train_titles)

train_titles_padded = np.array(pad_sequences(train_text_sequences, maxlen = max_length_title, \

                               padding=padding_type, truncating = trunc_type))

print(train_text[:2])

print(train_text_padded[:2])
# Tokenize validation texts and titles

val_text_sequences = tokenizer.texts_to_sequences(val_text)

val_text_padded = np.array(pad_sequences(val_text_sequences, maxlen = max_length, \

                               padding=padding_type, truncating = trunc_type))



val_titles_sequences = tokenizer.texts_to_sequences(val_titles)

val_titles_padded = np.array(pad_sequences(val_titles_sequences, maxlen = max_length_title, \

                               padding=padding_type, truncating = trunc_type))



# Tokenize test texts and titles

test_text_sequences = tokenizer.texts_to_sequences(test_text)

test_text_padded = np.array(pad_sequences(test_text_sequences, maxlen = max_length, \

                               padding=padding_type, truncating = trunc_type))



test_titles_sequences = tokenizer.texts_to_sequences(test_titles)

test_titles_padded = np.array(pad_sequences(test_text_sequences, maxlen = max_length_title, \

                               padding=padding_type, truncating = trunc_type))
print(val_titles_padded[0])

print(val_text_padded[0])

print(test_titles_padded[0])

print(test_text_padded[0])
# We're all set! Let's make the model

import tensorflow.keras.layers as layers

model = tf.keras.Sequential([

    layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # hyperparameters set above

    #layers.GlobalAveragePooling1D(),

    #layers.Dropout(0.5),

    layers.Bidirectional(layers.LSTM(32, activation='relu')),

    #layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# Let's prepare the data and train

train_data = (train_text_padded, train_titles_padded)

val_data = (val_text_padded, val_titles_padded)

num_epochs = 10

history = model.fit(train_data, train_truth, epochs=num_epochs, validation_data=(val_data, val_truth), verbose=2)
import matplotlib.pyplot as plt



# Some nice plots!



def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(history, "accuracy")

plot_graphs(history, "loss")
def predict(titles, texts, tokenizer=tokenizer, max_length=max_length, max_length_title=max_length_title, trunc_type=trunc_type, padding=padding_type):

    text_sequences = tokenizer.texts_to_sequences(texts)

    text_padded = pad_sequences(text_sequences, maxlen = max_length, \

                               padding=padding, truncating = trunc_type)

    titles_sequences = tokenizer.texts_to_sequences(titles)

    titles_padded = pad_sequences(titles_sequences, maxlen = max_length_title, \

                               padding=padding, truncating = trunc_type)

    predictions = model.predict((titles_padded, text_padded))

    return predictions
def accuracy(predictions, actual):

    assert len(predictions) == len(actual), "To compute accuracy, arrays must be same size"

    predictions = np.array(predictions)

    actual = np.array(actual)

    total_correct = 0.0

    for i in range(len(predictions)):

        if round(float(predictions[i])) == actual[i]:

            total_correct += 1

    return total_correct / len(predictions)
predictions = predict(test_titles, test_text)

print(accuracy(predictions, test_truth))

# nearly 80%, not too bad!
# I used a Kaggle dataset for this project, you can find it at

# https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset