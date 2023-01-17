# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import json

import urllib



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load data

df = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)



# we need only the text and target column

df = df[['headline', 'is_sarcastic']]



print(df.head())

print('\n ', df.shape)



text = np.array(df['headline'].values)

label = np.array(df['is_sarcastic'].values)
# split data into training and validation



train_text, validation_text, train_label, validation_label = train_test_split(text, label, test_size=0.1)
# define parameters for later use



pad_type = 'post' 

trunc_type = 'post'

word_count = 10000 # we will train our model with only top 10000 common words and rest will be excluded 

max_len = 32

threshold = 0.99 # to stop trainng the model once it reaches 90% accuracy

batch_size = 50
# tokenize sentences



tokens = tf.keras.preprocessing.text.Tokenizer(num_words= word_count, oov_token='<OOV>')

tokens.fit_on_texts(train_text)
# get word index and vectors for all sentences



word_index = tokens.word_index



train_seq = tokens.texts_to_sequences(train_text)

validation_seq = tokens.texts_to_sequences(validation_text)

# print a sentence and its sequence and label



print(train_text[1])

print(train_seq[1])

print(train_label[1])
# padd the sentence sequence for equal length



train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seq, padding = pad_type, truncating = trunc_type, maxlen = max_len)

validation_padded = tf.keras.preprocessing.sequence.pad_sequences(validation_seq, padding  = pad_type, truncating = trunc_type, maxlen = max_len)
# print padded sentence and sequence



print(train_padded[227])

print(train_text[227])

print(train_label[227])
print(train_padded.shape)

print(validation_padded.shape)
# create model 



model = tf.keras.Sequential([

    # add an embedding layer

    tf.keras.layers.Embedding(word_count, 16, input_length=max_len),

    

    # add the bi-lstm layer

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),

    

    # add another bi-lstm layer

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    

    # add dropout layer to prevent overfitting

    tf.keras.layers.Dropout(0.2),

    

    # add a dense layer

    tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),

    

    # add the prediction layer

    tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid),

])



model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])



model.summary()
# define a callback function to stop training when thrshold is reached



class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if logs.get('accuracy') > threshold:

            self.model.stop_training = True

            print(' \n Terminating training as model reached ' + str(threshold * 100) + ' % accuracy \n')



callback_func = myCallback()
history = model.fit(train_padded, train_label, validation_data=(validation_padded, validation_label), epochs = 100, batch_size=batch_size, callbacks = [callback_func], verbose=1)
# plot loss



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['Training loss', 'Validation loss'])

plt.show()
# plot accuracy 

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['Training accuracy', 'Validation accuracy'])

plt.show()
# extract embedding

e = model.layers[0]

weights = e.get_weights()[0]

print(weights.shape)
# map embeddings to words



reverse_word_index = {v:k for k,v in word_index.items()}



words = []

embeddings = []



for w in range(1, word_count):

    

    words.append(reverse_word_index[w])

    embeddings.append(weights[w])

    

words = np.array(words)

embeddings = np.array(embeddings)



print(words.shape)

print(embeddings.shape)
print(words[88])

print(embeddings[88])
# export for mapping



words = pd.DataFrame(words)

embeddings = pd.DataFrame(embeddings)



words.to_csv('meta.tsv', sep='\t', index=False)

embeddings.to_csv('vecs.tsv', sep='\t', index=False)
"""

visit this url for an ineractive visualisation 



https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/rmishra258/ef1518e16c0461c3c98120e3c5dfdaa7/raw/322c369a328f0b05b68fea3346ed7d00fa70af72/template_projector_config.json



please consider upvoting

"""