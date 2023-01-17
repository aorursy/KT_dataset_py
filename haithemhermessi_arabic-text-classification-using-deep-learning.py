!pip install utils
!pip install tensorflow --upgrade
#!pip install  unicode
#Import libraries
import os
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import Callback
from gensim.models.keyedvectors import KeyedVectors
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from keras.layers import Embedding, Dense, Dropout, Input#, LSTM, Bidirectional
from keras.layers import MaxPooling1D, Conv1D, Flatten, LSTM
from keras.preprocessing import sequence#, text
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import text
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix as confmat,
    classification_report as creport
)
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print(tf.__version__)
#Load dataset using the new tensorflow text_dataset_from_directory 

seed=42
data_paths = '../input/sanad-dataset'
labels=os.listdir(data_paths) 
raw_data_train = tf.keras.preprocessing.text_dataset_from_directory(
    data_paths,
    labels="inferred",
    label_mode="int",
    #class_names=classes,
    #batch_size=1,
    max_length=None,
    shuffle=True,
    seed=seed,
    validation_split=None,
    subset=None,
    follow_links=False,
)
raw_data_test = tf.keras.preprocessing.text_dataset_from_directory(
    data_paths, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)
x_train=[]
y_train=[]
for text_batch, label_batch in raw_data_train:
    for i in range(len(text_batch)):
        s=text_batch.numpy()[i].decode("utf-8") 
        x_train.append(s)
        y_train.append(raw_data.class_names[label_batch.numpy()[i]])
        #print(label_batch.numpy()[i])
print(len(x_train))
print(len(y_train))
x_test=[]
y_test=[]
for text_batch, label_batch in raw_data_test:
    for i in range(len(text_batch)):
        s=text_batch.numpy()[i].decode("utf-8") 
        x_test.append(s)
        y_test.append(raw_data.class_names[label_batch.numpy()[i]])
        #print(label_batch.numpy()[i])
print(len(x_test))
print(len(y_test))
#To prevent train/test skew (also know as train/serving skew), it is important to preprocess the data identically
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')
max_features = 10000
sequence_length = 500

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
# Make a text-only dataset (without labels), then call adapt
train_text = raw_data_train.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
# Standardize, tokenize, and vectorize our data
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label
# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_data_train))
first_review, first_label = text_batch[0], label_batch[0]
print("Text", first_review)
print("Label", raw_data_train.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

train_ds = raw_data_train.map(vectorize_text)
val_ds = raw_data_test.map(vectorize_text)

embedding_dim = 100
max_length = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim,  trainable=False),
  layers.Dropout(0.2),
  layers.Conv1D(64, 5, activation='relu'),
  layers.MaxPooling1D(pool_size=4),
  layers.LSTM(64),
  layers.Dense(1, activation='softmax')])

model.summary()
model.compile(loss=losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
epochs = 10
history = model.fit(
    train_ds,
    batch_size=batch_size,
    validation_data=val_ds,
    epochs=epochs)
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")