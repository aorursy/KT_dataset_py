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
sentences = []

labels = []

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

print(len(stopwords))
import csv

import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('../input/amazon-fine-food-reviews/Reviews.csv','r') as csvfile:

    reader = csv.reader(csvfile, delimiter = ',')

    next(reader)

    for row in reader:

        labels.append(row[6])

        sentence = row[9]

        for word in stopwords:

            token = ' '+word+' '

            sentence.replace(token,' ')

            sentence.replace('  ',' ')

        sentences.append(sentence)



print(len(labels))

print(len(sentences))

print(sentences[0])
vocab_size = 10000

embedding_dim = 16

max_length = 120

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_tok)

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences[0])
padded = pad_sequences(sequences, padding = padding_type, maxlen = max_length,truncating = trunc_type)
print(padded[0])

print(type(padded))

print(type(labels))
labels = np.array(labels)

print(type(labels))
label_tokenizer = Tokenizer()

label_tokenizer.fit_on_texts(labels)
word_index_labels = label_tokenizer.word_index

print(word_index_labels)
label_seq = label_tokenizer.texts_to_sequences(labels)

label_padd = pad_sequences(label_seq)
print(label_padd[0])
from keras.models import Model 

from keras.layers import *

from keras.utils.vis_utils import plot_model

from keras.models import Sequential
max_features = vocab_size

maxlen = max_length

filters = 250

kernel_size = 3

hidden_dims = 250



model = Sequential()

model.add(Embedding(max_features,embedding_dim,input_length=maxlen))

model.add(Dropout(0.2))

model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))

model.add(Dropout(0.2))

model.add(Activation('relu'))



# We project onto a single unit output layer, and squash it with a sigmoid:

model.add(Dense(6))

model.add(Activation('softmax'))

# compile 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary() 
plot_model(model, show_shapes=True, to_file='model.png') 
num_epochs = 5

history = model.fit(padded, label_padd, epochs=num_epochs, verbose=1, validation_split=0.3)
import matplotlib.pyplot as plt



fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

fig.suptitle("Performance of Model without pretrained embeddings")

ax1.plot(history.history['accuracy'])

ax1.plot(history.history['val_accuracy'])

vline_cut = np.where(history.history['val_accuracy'] == np.max(history.history['val_accuracy']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Model Accuracy")

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Model Loss")

ax2.legend(['train', 'test'])

plt.show()