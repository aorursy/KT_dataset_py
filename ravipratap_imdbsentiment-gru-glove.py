 # The model is for Sentiment classification on IMBD dataset.
 # It is based on neural network - GRU RNN and pre-trained GloVe word embeddings.
 # It uses tensorflow keras APIs.
    
 # Steps:
 # Step1. load corpus data, pre-process and split into train and test dataset. [pending HTML tags removal]
 # Step2. create a vocabulary index on corpus. tokenize and vectorize corpus data.
 # Step3. load pre-trained GloVe word embeddings.
 # Step4. build and evaluate model on training, validation and test data.
#import required libraries

import numpy as np
import pandas as pd
import os
import pathlib
import codecs
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, GRU, LSTM
print(tf.__version__)
#import dataset and analyze dataset

df = pd.read_csv('../input/movie-review/labelled_full_dataset.csv')
df.isnull().values.any()
df.shape
df.head()
import seaborn as sns
sns.countplot(x='label', data=df)
#view one sample

df['review'][1]
# #splitting training data into validation and training

#limiting data to less values to run in less then 5 minutes
x_train = df.loc[:9999, 'review'].values
y_train = df.loc[:9999, 'label'].values
x_test = df.loc[10000:12499, 'review'].values
y_test = df.loc[10000:12499, 'label'].values

# #total data split
# x_train = df.loc[:39999, 'review'].values
# y_train = df.loc[:39999, 'label'].values
# x_test = df.loc[40000:, 'review'].values
# y_test = df.loc[40000:, 'label'].values
print(len(x_train), len(x_test), len(y_train), len(y_test))
x_train[1]
#create a vocabulary index

tokenizer = tf.keras.preprocessing.text.Tokenizer()
#vocabulary = tokenizer.fit_on_texts(df['review'])
vocabulary = tokenizer.fit_on_texts(x_train) 
#running tokenizer on x_train only for limiting values, else would be run on raw text data
print(tokenizer)
print(vocabulary)
#define vocabulary size

vocabulary_size_max = len(tokenizer.word_index) + 1
print(vocabulary_size_max)
#max length for padding

#max_length = max([len(s.split()) for s in df['review']])
max_length = max([len(s.split()) for s in x_train]) 
#limiting padding to x_train data else would be run on raw text data
print(max_length)
#vectorize tokens

x_train_vector = tokenizer.texts_to_sequences(x_train)
x_test_vector = tokenizer.texts_to_sequences(x_test)
print("train vector is:", x_train_vector[1])
print("test vector is:", x_test_vector[1])
#pad sequences
#sequences shorter than the length are padded in the beginning and 
#sequences longer are truncated at the beginning.

x_train_pad = pad_sequences(x_train_vector, maxlen = max_length, padding = 'post')
x_test_pad = pad_sequences(x_test_vector, maxlen = max_length, padding = 'post')
print("train padding is:", x_train_pad[1])
print("test padding is:", x_test_pad[1])
#tokenizer.word_index.items()
#load pre-trained word embedding in a dictionary
#dictionary with key = word and value = embedding in the file

glove_file = '../input/glove6b50dtxt/glove.6B.50d.txt'
embedding_dict = {}
glove = codecs.open(glove_file, encoding = 'utf8')
for line in glove:
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_dict[word] = coef
glove.close()
print(embedding_dict["the"])
#embedding matrix with ONLY the words present in the input vocabulary i.e. corpus and
#their corresponding embedding vector
#vocab_size = len(token.word_index)+1
#shape of embedding matrix: vocabulary_size_max, glove_dimension

embedding_matrix = np.zeros((vocabulary_size_max,50))
for word,i in tokenizer.word_index.items():
    embedding_value = embedding_dict.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value
len(embedding_matrix), embedding_matrix.size
print(embedding_matrix[1])
#load the pre-trained word embeddings matrix into an Embedding layer
# Note that we set trainable=False so as to keep the embeddings fixed
# (we don't want to update them during training).

embedding_dim = 50 #dimensions of embedding layer

from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    input_dim = vocabulary_size_max,
    output_dim = embedding_dim,
    input_length = max_length,
    embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix),
    trainable = False
)
#build model
#GRU default with tanh activation, recurrent activation default sigmoid

model = Sequential()
model.add(embedding_layer)
model.add(GRU(units = 16, dropout = 0.2, recurrent_dropout = 0.2, activation = 'tanh'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
# # model to learn new word embeddings and NOT use pre-trained
# #build model
# #GRU default with tanh activation, recurrent activation default sigmoid
# model = Sequential()
# model.add(Embedding(input_dim = vocabulary_size_max, output_dim = embedding_dim, input_length = max_length))
# model.add(GRU(units = 16, dropout = 0.2, recurrent_dropout = 0.2, activation = 'tanh'))
# model.add(Dense(1, activation = 'sigmoid'))
# model.summary()
#compile model

model.compile(optimizer = 'adam', loss = tf.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
#train model
#training on less epochs to run it in less time. ideal would be to increase epochs.

history = model.fit(x = x_train_pad, y = y_train, batch_size = 512, epochs = 5, validation_split = 0.25, verbose = 2)
# plot loss and accuracy of training and validation

history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
#evaluate the model

evaluate = model.evaluate(x = x_test_pad, y = y_test, batch_size = 512, verbose = 1, return_dict = True)
evaluate.keys()
test_acc = evaluate['accuracy']
test_loss = evaluate['loss']
print("Test loss: ", test_loss*100, '%')
print("Test accuracy: ", test_acc*100, '%')
