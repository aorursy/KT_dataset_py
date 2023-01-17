!pip install num2words
import tensorflow as tf
from keras import layers
import keras
import matplotlib.pyplot as plt

import nltk, re
import os
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer       
from num2words import num2words

from tqdm import tqdm
import pandas as pd
import random
from collections import Counter
import tensorflow_hub as hub

import numpy as np
from sklearn.model_selection import train_test_split   

plt.style.use('fivethirtyeight') 
%matplotlib inline
def remove_url_and_symbols(comment):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",comment).split())

def lower_comment(comment):
    return comment.lower()


def tokenize_comment(comment):
    return comment.split()

def remove_stop_words(comment_tokenized):
    stopwords_english = stopwords.words('english')
    comment_tokenized_cleaned = []
    for word in comment_tokenized:
        if(word not in stopwords_english):
            comment_tokenized_cleaned.append(word)
    return comment_tokenized_cleaned

def convert_numbers_to_words(comment_tokenized):
    comment_tokenized_cleaned = []
    for word in comment_tokenized:  # Go through every word in your tokens list
        try:
            comment_tokenized_cleaned.append(num2words(word))
        except:
            comment_tokenized_cleaned.append(word)
    return comment_tokenized_cleaned

def remove_ponctuation_from_tokenized(comment_tokenized):
    comment_tokenized_cleaned = []
    for word in comment_tokenized:
        comment_tokenized_cleaned.append(remove_url_and_symbols(word))
    
    return comment_tokenized_cleaned

def stem_words(comment_tokenized):
    stemmer = PorterStemmer()
    comment_tokenized_cleaned = []
    for word in comment_tokenized:  # Go through every word in your tokens list
        comment_tokenized_cleaned.append(stemmer.stem(word))   # stemming word
    return comment_tokenized_cleaned
def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s) for s in sample_texts]
    return np.median(num_words)
def get_optimizer(batch_size_var, X_train):
    # Many models train better if you gradually reduce the learning rate during training. 
    # Use optimizers.schedules to reduce the learning rate over time
    N_TRAIN = X_train.shape[0]
    STEPS_PER_EPOCH = N_TRAIN//batch_size_var

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, 
                                                                 decay_steps=STEPS_PER_EPOCH*1000, 
                                                                 decay_rate=1, 
                                                                 staircase=False)
    
    return tf.keras.optimizers.Adam(lr_schedule)

def compile_model(model, loss_func, batch_size_var, epochs_var, X_train, Y_train, X_test, Y_test):
  model.compile(optimizer=get_optimizer(batch_size_var, X_train), 
                loss=loss_func,
                metrics=['accuracy'])

  history = model.fit(
      X_train,
      Y_train,
      batch_size=batch_size_var,
      epochs=epochs_var,
      # We pass some validation for
      # monitoring validation loss and metrics
      # at the end of each epoch
      validation_data=(X_test, Y_test),
  )


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
  return model
read_data = pd.read_csv("../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv", encoding = "ISO-8859-1")
data = pd.DataFrame()
data['name'] = list(read_data['name'])
data['text'] = list(read_data['text'])
data['gender'] = list(read_data['gender'])
data
list(data['text'])[0]
# Clean the data.
cleaned_comments = []
for i in tqdm(range(len(list(data['text'])))):
    text = remove_url_and_symbols(list(data['text'])[i])
    text = lower_comment(text)
    text_tokenized = tokenize_comment(text)
    text_tokenized = convert_numbers_to_words(text_tokenized)
    text_tokenized = remove_stop_words(text_tokenized)
    text_tokenized = remove_ponctuation_from_tokenized(text_tokenized)
    text_tokenized = stem_words(text_tokenized)
    
    cleaned_comments.append(text_tokenized)
# Clean the data.
cleaned_names = []
for i in tqdm(range(len(list(data['name'])))):
    name = lower_comment(list(data['name'])[i])
    cleaned_names.append(name)
# Clean the data.
cleaned_gender = []
for i in tqdm(range(len(list(data['gender'])))):
    if(list(data['gender'])[i]=='male'):
      cleaned_gender.append(0)
    else:
      cleaned_gender.append(1)
get_num_words_per_sample(cleaned_comments)   # Median number of words per sample.
vocabulary_size = 20000
text_length = 8
len(list(data['name'])) / get_num_words_per_sample(cleaned_comments)
cleaned_comments_sentance = []
for sentance in cleaned_comments:
    ch = ""
    for word in sentance:
        ch = ch + word + " "
    cleaned_comments_sentance.append(ch)
# num_words: The maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
tokenizer_text = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocabulary_size, 
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789', 
    lower=True, 
    split=" ", 
    char_level=False
)
tokenizer_text.fit_on_texts(cleaned_comments_sentance)
sequences_tokenizer_text = tokenizer_text.texts_to_sequences(cleaned_comments_sentance)
# num_words: The maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
tokenizer_name = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocabulary_size, 
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789', 
    lower=True, 
    split=" ", 
    char_level=False
)
tokenizer_name.fit_on_texts(cleaned_names)
sequences_tokenizer_name = tokenizer_name.texts_to_sequences(cleaned_names)
for i in tqdm(range(len(sequences_tokenizer_text))):
  if(len(sequences_tokenizer_text[i])>8):
    sequences_tokenizer_text[i] = sequences_tokenizer_text[i][:8]

  if(len(sequences_tokenizer_text[i])<8):
    for j in range(8-len(sequences_tokenizer_text[i])):
      sequences_tokenizer_text[i].append(0)
  
for i in tqdm(range(len(sequences_tokenizer_name))):
  if(len(sequences_tokenizer_name[i])>8):
    sequences_tokenizer_name[i] = sequences_tokenizer_name[i][:8]

  if(len(sequences_tokenizer_name[i])<8):
    for j in range(8-len(sequences_tokenizer_name[i])):
      sequences_tokenizer_name[i].append(0)
input_layer_text = keras.Input(shape=(None,), name="Input_text", dtype=tf.int64)  
embedding_layer_text = layers.Embedding(vocabulary_size, 64)(input_layer_text)
x_text = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_layer_text)
x_text = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2)(x_text)

input_layer_name = keras.Input(shape=(None,), name="Input_name", dtype=tf.int64)  
embedding_layer_name = layers.Embedding(vocabulary_size, 64)(input_layer_name)
x_name = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_layer_name)
x_name = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2)(x_name)

x = layers.concatenate([x_text, x_name])

output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=[input_layer_text, input_layer_name], outputs=output_layer, name="model")

model.summary()
keras.utils.plot_model(model, "model.png", show_shapes=True)
X_text = pd.DataFrame(sequences_tokenizer_text)
X_name = pd.DataFrame(sequences_tokenizer_name)
Y = pd.DataFrame(cleaned_gender)

X_train_text = X_text[:15000]
X_test_text = X_text[15000:]
X_train_name = X_name[:15000]
X_test_name = X_name[15000:]
Y_train = Y[:15000]
Y_test = Y[15000:]
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=2)
def get_optimizer(batch_size_var, X_train):
    # Many models train better if you gradually reduce the learning rate during training. 
    # Use optimizers.schedules to reduce the learning rate over time
    N_TRAIN = X_train.shape[0]
    STEPS_PER_EPOCH = N_TRAIN//batch_size_var

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, 
                                                                 decay_steps=STEPS_PER_EPOCH*1000, 
                                                                 decay_rate=1, 
                                                                 staircase=False)
    
    return tf.keras.optimizers.Adam(lr_schedule)

def compile_model(model, loss_func, batch_size_var, epochs_var, X_train1, X_train2, Y_train, X_test1, X_test2, Y_test):
  model.compile(optimizer=get_optimizer(batch_size_var, X_train1), 
                loss=loss_func,
                metrics=['accuracy'])


  history = model.fit(
      x=[X_train1, X_train2], y=Y_train,
      validation_data=([X_test1, X_test2], Y_test),
      batch_size=batch_size_var,
      epochs=epochs_var
    )


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
  return model
model = compile_model(model, 'categorical_crossentropy', 40, 10, X_train_text, X_train_name, Y_train, X_test_text, X_test_name, Y_test)
X_text2 = pd.DataFrame(cleaned_comments_sentance)
X_name2 = pd.DataFrame(cleaned_names)
Y2 = pd.DataFrame(cleaned_gender)

X_train_text2 = X_text2[:15000]
X_test_text2 = X_text2[15000:]
X_train_name2 = X_name2[:15000]
X_test_name2 = X_name2[15000:]
Y_train2 = Y2[:15000]
Y_test2 = Y2[15000:]
Y_train2 = tf.keras.utils.to_categorical(Y_train2, num_classes=2)
Y_test2 = tf.keras.utils.to_categorical(Y_test2, num_classes=2)
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
input_layer_text = keras.Input(shape=(), name="Input_text", dtype=tf.string)  

hub_layer_text = hub.KerasLayer(embedding, trainable=True, name='embedding_text')(input_layer_text)
x_text = tf.expand_dims(hub_layer_text, 1)
x_text = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x_text)
x_text = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x_text)
x_text = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2)(x_text)

input_layer_name = keras.Input(shape=(), name="Input_name", dtype=tf.string)  
hub_layer_name = hub.KerasLayer(embedding, trainable=True, name='embedding_name')(input_layer_name)
x_name = tf.expand_dims(hub_layer_name, 1)
x_name = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x_name)
x_name = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x_name)
x_name = layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2)(x_name)

x = layers.concatenate([x_text, x_name])

output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

second_model = keras.Model(inputs=[input_layer_text, input_layer_name], outputs=output_layer, name="model")

second_model.summary()
keras.utils.plot_model(second_model, "model.png", show_shapes=True)
second_model = compile_model(second_model, 'categorical_crossentropy', 40, 10, X_train_text2, X_train_name2, Y_train2, X_test_text2, X_test_name2, Y_test2)