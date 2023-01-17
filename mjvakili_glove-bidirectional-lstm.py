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
import numpy as np 

import re  #regular expression

from bs4 import BeautifulSoup

import pandas as pd 

from sklearn import model_selection, preprocessing

import os

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, Dropout, Bidirectional

from tensorflow.keras.models import Model

from tensorflow.keras.initializers import Constant

from tensorflow.keras import metrics

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
print('Indexing word vectors.')

#Many thanks to rtatman for hosting the GloVe word embeddings dataset on Kaggle

#https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation

GLOVE_DIR = '/kaggle/input/glove-global-vectors-for-word-representation/'

embeddings_index = {}

with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:

    for line in f:

        word, coefs = line.split(maxsplit=1)

        coefs = np.fromstring(coefs, 'f', sep=' ')

        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
print('Reading the datasets')

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print('Colimns of the training and test datasets are:')

print(train_df.keys())

print(test_df.keys())
url_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

# Thanks to l3nnys for this nice tutorial on text preprocessing

#https://www.kaggle.com/l3nnys/useful-text-preprocessing-on-the-datasets



def remove_html(text):

    '''

    remove the HTML tags and URLS from the tweets

    '''

    if text:

        # BeautifulSoup on content

        soup = BeautifulSoup(text, "html.parser")

        # Stripping all <code> tags with their content if any

        if soup.code:

            soup.code.decompose()

        # Get all the text out of the html

        text =  soup.get_text()

        # Returning text stripping out all uris

        return re.sub(url_re, "", text)

    else:

        return ""

  

train_df['text'] = train_df['text'].map(lambda x: remove_html(x))

test_df['text'] = test_df['text'].map(lambda x: remove_html(x))
# Removing emojis thanks to this instruction on stackoverflow:

#https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python



def remove_emojis(text):

  emoji_pattern = re.compile("["

        r"\U0001F600-\U0001F64F"  # emoticons

        r"\U0001F300-\U0001F5FF"  # symbols & pictographs

        r"\U0001F680-\U0001F6FF"  # transport & map symbols

        r"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           "]+", flags=re.UNICODE)

  text = emoji_pattern.sub(r'', text)# no emoji 

  

  return text



example_text = r'This dog is funny \U0001f602'

train_df['text'] = train_df['text'].map(lambda x: remove_emojis(x))

test_df['text'] = test_df['text'].map(lambda x: remove_emojis(x))
#!pip install pyspellchecker
"""

def spell_correct(text):

  '''

     check the spellingd with pyspellchecker 

     and replaces the words with incorrect spellings 

     with the most likely correctly spelled candidate

  '''  

  text_correct = [spell.correction(x) for x in text.split()]

  text_correct = " ".join(str(x) for x in text_correct)

  

  return text_correct



easy_example_text = r'What is hapening in the forrestt?'

difficult_example_text = r'What is the prive of libery?'

print(spell_correct(easy_example_text))

print(spell_correct(difficult_example_text))



#Now let's correct the spellings of words in the training and the test set

train_df['text'] = train_df['text'].map(lambda x: spell_correct(x))

test_df['text'] = test_df['text'].map(lambda x: spell_correct(x))

"""
vocab_size = 10000

embedding_dim = 100

max_length = 50

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

validation_split = 0.3



tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(train_df.text)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



training_sequences = tokenizer.texts_to_sequences(train_df.text)

training_padded = pad_sequences(training_sequences, maxlen = max_length, 

                                padding = padding_type, truncating = trunc_type)

print('Shape of the data vector is', training_padded.shape, train_df.target.shape)
print('Preparing the embedding matrix')

num_words = min(vocab_size, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embedding_dim))

for word, index in word_index.items():

  if index >= vocab_size:

    continue

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:

    embedding_matrix[index] = embedding_vector
embedding_layer = Embedding(num_words, embedding_dim, 

                           embeddings_initializer = Constant(embedding_matrix), 

                           input_length = max_length, 

                           trainable = False)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(training_padded, 

                                                                      train_df.target, 

                                                                      test_size = validation_split, 

                                                                      random_state=1)
#Copied this from this example on tensorflow website

#https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

METRICS = [

      metrics.TruePositives(name='tp'),

      metrics.FalsePositives(name='fp'),

      metrics.TrueNegatives(name='tn'),

      metrics.FalseNegatives(name='fn'), 

      metrics.BinaryAccuracy(name='accuracy'),

      metrics.Precision(name='precision'),

      metrics.Recall(name='recall'),

      metrics.AUC(name='auc')]



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc', 

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True)
#Defining a Sequential Keras model with Convolution & Global Max Pooling

sequence_input = Input(shape = (max_length, ))

embedded_sequences = embedding_layer(sequence_input)

x = Bidirectional(tf.keras.layers.LSTM(32))(embedded_sequences)

#x = Conv1D(32, 5, activation='relu')(embedded_sequences)

x = Dropout(0.5)(x)

x = Dense(24, activation = 'relu')(x)

x = Dropout(0.5)(x)

output = Dense(1, activation = 'sigmoid')(x)

model =  Model(sequence_input, output)

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr = .0002) ,metrics = METRICS)

history = model.fit(X_train, y_train, batch_size = 64, epochs = 30, 

                    callbacks = [early_stopping],

                    validation_data = (X_valid, y_valid))
def plot_model_eval(history):

  '''

  a simple funtion for model evaluation 

  according to different metrics

  '''  

  string = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'tp']  

  cnt = 0

  ncols, nrows = 3, 2  

  fig = plt.figure(constrained_layout=True, figsize = (10,10))

  gs = gridspec.GridSpec(ncols = 3, nrows = 2, figure = fig)

  for i in range(nrows):

    for j in range(ncols):

      ax = plt.subplot(gs[i,j]) 

      ax.plot(history.history[string[cnt]])

      ax.plot(history.history['val_'+string[cnt]]) 

      ax.set_xlabel("Epochs")

      ax.set_ylabel(string[cnt])

      ax.legend([string[cnt], 'val_'+string[cnt]])

      cnt +=1

        

plot_model_eval(history)
#Now let's look at the confusion matrix

pred_valid = model.predict(X_valid)

pred_valid = np.round(pred_valid).astype(int)

confusion_matrix(y_valid, pred_valid)
test_sequences = tokenizer.texts_to_sequences(test_df.text)

test_padded = pad_sequences(test_sequences, maxlen = max_length, 

                                padding = padding_type, truncating = trunc_type)

print('Shape of the data vector is', test_padded.shape)
predictions = model.predict(test_padded)

predictions = np.round(predictions).astype(int).flatten()
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = predictions.astype(int)

print(sample_submission.head())

sample_submission.to_csv("submission.csv", index=False)