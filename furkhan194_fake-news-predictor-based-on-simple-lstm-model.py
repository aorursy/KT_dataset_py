# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import Callback



from sklearn.model_selection import train_test_split



import string



from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))



import plotly

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
legit, fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv'), pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

legit.sample(10)
legit['target'] = 1

fake['target'] = 0

data = pd.concat([legit, fake], axis=0)

data.sample(10)
data.isnull().sum()
# Hyperparameters for title and text

vocab_size = 100000

embedding_dim_title = 128

max_length_title = 40

embedding_dim_text = 500

max_length_text = 500

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>'

test_ratio = .3

embedding_dim = 500

max_length_text = 500
# detect and init the TPU

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)



    # instantiate a distribution strategy

    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    TPU_EXIST = True

except Exception as e:

    print(e)

    TPU_EXIST = False
# Text cleaning

def clean(text):

    #1. Remove punctuation

    translator1 = str.maketrans(string.punctuation, ' '*len(string.punctuation))

    text = text.translate(translator1)

    

    #2. Convert to lowercase characters

    text = text.lower()

    

    #3. Remove stopwords

    text = ' '.join([word for word in text.split() if word not in STOPWORDS])

    

    return text
# Apply cleaning to title and text in dataset

data['title'] = data['title'].apply(clean)

data['text'] = data['text'].apply(clean)

data.sample(10)
def preprocessing(data, dependent_column=None, target='target', max_len=40):

    train_X, test_X, train_y, test_y = train_test_split(data[dependent_column], data[target], test_size=test_ratio)

    tokenizer = Tokenizer(num_words=vocab_size,

                          oov_token=oov_tok)

    tokenizer.fit_on_texts(train_X)

    train_sequences = tokenizer.texts_to_sequences(train_X)

    train_padded = pad_sequences(train_sequences, maxlen=max_len,

                                padding=padding_type,

                                truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(test_X)

    test_padded = pad_sequences(test_sequences, maxlen=max_len,

                               padding=padding_type,

                               truncating=trunc_type)

    return train_padded, test_padded, train_y, test_y
# Create the model

def model_creation(vocab_size=vocab_size, embedding_dim=128):

    if TPU_EXIST:

        with tpu_strategy.scope():

            model = tf.keras.Sequential()

            model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))

            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))

            model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))

            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    else:

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))

        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))

        model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model
def train_model(model, train_X, train_Y, test_X, test_Y, epochs):

    class CustomCallback(Callback):

        def on_epoch_end(self, epoch, logs={}):

            if logs.get('acc') > 0.99:

                print(f'Accuracy reached {logs.get("acc")*100:0.2f}. Stopping the training')

                self.model.stop_training = True



    history = model.fit(train_X, train_Y,

                       epochs=epochs,

                       batch_size=64,

                       validation_data=[test_X, test_Y],

                       callbacks=[CustomCallback()])

    return history
train_padded, test_padded, train_y, test_y = preprocessing(data, dependent_column='title', max_len=max_length_title)

model = model_creation(embedding_dim=embedding_dim_title)

history_title = train_model(model, train_padded, train_y, test_padded, test_y, 15)
train_padded, test_padded, train_y, test_y = preprocessing(data, dependent_column='text', max_len=max_length_title)

model = model_creation(embedding_dim=embedding_dim_text)

history_text = train_model(model, train_padded, train_y, test_padded, test_y, 15)
title_max_acc = max(history_title.history.get('acc'))

text_max_acc = max(history_text.history.get('acc'))



fig = go.Figure()

fig.add_trace(go.Scatter(x=['Title', 'Text'],

                        y=[title_max_acc,

                          text_max_acc],

                        mode='lines+markers',

                        name='Accuracies of Models'))

fig.update_layout(title='Accuracies Differences',

                 xaxis_title='Case Name',

                 yaxis_title='Accuracy of Model')

fig.show()