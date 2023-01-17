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
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
data=pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json',lines=True)
print(data.head())

print('Length of data: %d'%(len(data)))
split_num1,split_num2=int(len(data)*0.8),int(len(data)*0.9)

train=data[:split_num1]

valid=data[split_num1:split_num2]

test=data[split_num2:]

print('Length of train_set: %d'%(len(train)),'\n',

      'Length of valid_set: %d'%(len(valid)),'\n',

      'Length of test_set: %d'%(len(test)))
list(train['headline'].values)[:10]
vocab_size=1000

embedding_dim=16

max_length=16

trunc_type='post'

pad_type='post'

oov_tok='<OOV>'



train_sentences=list(train['headline'].values)

valid_sentences=list(valid['headline'].values)

test_sentences=list(test['headline'].values)

train_labels=np.array(train['is_sarcastic'].values)

valid_labels=np.array(valid['is_sarcastic'].values)

test_labels=np.array(test['is_sarcastic'].values)



tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(train_sentences)

word2idx=tokenizer.word_index

train_sequences=tokenizer.texts_to_sequences(train_sentences)

padded=pad_sequences(train_sequences,maxlen=max_length,padding=pad_type,truncating=trunc_type)

valid_sequences=tokenizer.texts_to_sequences(valid_sentences)

valid_padded=pad_sequences(valid_sequences,maxlen=max_length,padding=pad_type,truncating=trunc_type)

test_sequences=tokenizer.texts_to_sequences(test_sentences)

test_padded=pad_sequences(test_sequences,maxlen=max_length,padding=pad_type,truncating=trunc_type)



print(padded[0],'\n',padded.shape)





model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),# tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 30

history=model.fit(padded, train_labels, epochs=num_epochs, validation_data=(valid_padded, valid_labels),verbose=2)
import matplotlib.pyplot as plt

def plot_graph(history,string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel('Epochs')

    plt.ylabel(string)

    plt.legend([string,'val_'+string])

    plt.show()

plot_graph(history,'accuracy')

plot_graph(history,'loss')