## Import basic packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Read data

train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip',sep="\t") 

test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep="\t") 
train.head()
test.head()
train.Sentiment.value_counts()
train.info()
## Show the number of class distributed

plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(x=train.Sentiment,data=train)
import string

string.punctuation
train['Phrase1']=train.Phrase.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)).lower())

test['Phrase1']=test.Phrase.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)).lower())
train.head()
train_shuffle=train.sample(frac=1, random_state=1)
split=int(0.7*train.shape[0])

train_data=train_shuffle[0:split]

valid_data=train_shuffle[split:]
words=set()

word_to_vec_map={}

with open('/kaggle/input/glove-50d/glove.6B.50d.txt','r',encoding='UTF-8') as f:

    for line in f:

        value=line.strip().split()

        word=value[0]

        words.add(word)

        word_to_vec_map[word]=np.array(value[1:],dtype=np.float64)
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers

from tensorflow import keras
## fit train data into vector

tokenizer=Tokenizer()

tokenizer.fit_on_texts(train.Phrase1)

train_sequences=tokenizer.texts_to_sequences(train_data.Phrase1)

valid_sequences=tokenizer.texts_to_sequences(valid_data.Phrase1)

test_sequences=tokenizer.texts_to_sequences(test.Phrase1)
## Pad sequence data with 0 for same length

maxlen=train.Phrase1.apply(lambda x: len(x)).max()

train_padded=pad_sequences(train_sequences,maxlen=maxlen,padding='post',truncating='post')

valid_padded=pad_sequences(valid_sequences,maxlen=maxlen,padding='post',truncating='post')

test_padded=pad_sequences(test_sequences,maxlen=maxlen,padding='post',truncating='post')

print(train_padded.shape,valid_padded.shape,test_padded.shape)
word_index=tokenizer.word_index

vocab_size=len(word_index)

embedding_matrix=np.zeros((vocab_size+1,50))   ## for unknown word, add 1 in vocab_size
## Use Glove wordembedding

for word, i in word_index.items():

    embedding_vector=word_to_vec_map.get(word)

    if embedding_vector is not None:

        embedding_matrix[i]=embedding_vector
train_x=np.array(train_padded)

train_y=np.array(train_data.Sentiment)

valid_x=np.array(valid_padded)

valid_y=np.array(valid_data.Sentiment)

test_x=np.array(test_padded)
N_TRAIN=len(train_x)

BATCH_SIZE=256

STEPS_PER_EPOCH=N_TRAIN/BATCH_SIZE



## decrease the learning rate when epoch increase.

lr_decay=tf.keras.optimizers.schedules.InverseTimeDecay(

    1e-2,

    decay_steps=STEPS_PER_EPOCH*100,

    decay_rate=1,

    staircase=False

)

histories={}
## Simple NN model

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()
histories['ANN']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 128)
colors=['blue','green']

def plot_metrics(history):

  metrics =  ['loss', 'accuracy']

  for n, metric in enumerate(metrics):

    name = metric.replace("_"," ").capitalize()

    plt.subplot(1,2,n+1)

    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')

    plt.plot(history.epoch, history.history['val_'+metric],

             color=colors[1], linestyle="--", label='Val')

    plt.xlabel('Epoch')

    plt.ylabel(name)

    plt.legend()
plt.figure(figsize=(20,7))

plot_metrics(histories['ANN'])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv1D(128, 2, padding='same',activation='relu'),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()
histories['CNN']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 256)
plt.figure(figsize=(20,7))

plot_metrics(histories['CNN'])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(256, return_sequences=True),

    tf.keras.layers.LSTM(256),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(125, activation='relu'),

    #tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()

histories['LSTM_1']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 128)
plt.figure(figsize=(20,7))

plot_metrics(histories['LSTM_1'])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()
histories['LSTM_2']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 256)
plt.figure(figsize=(20,7))

plot_metrics(histories['LSTM_2'])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()
histories['LSTM_3']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 256)
plt.figure(figsize=(20,7))

plot_metrics(histories['LSTM_3'])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()
histories['LSTM_4']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 256)
plt.figure(figsize=(20,7))

plot_metrics(histories['LSTM_4'])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,50,input_length=maxlen, weights=[embedding_matrix],trainable=False),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr_decay),metrics='accuracy')

model.summary()
histories['LSTM_5']=model.fit(train_x, train_y, epochs = 10,validation_data=(valid_x,valid_y), batch_size = 256)
plt.figure(figsize=(20,7))

plot_metrics(histories['LSTM_5'])
test_y=model.predict_classes(test_x)
sub_file = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv',sep=',')

sub_file.Sentiment=test_y

sub_file.to_csv('Submission.csv',index=False)