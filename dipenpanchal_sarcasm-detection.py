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
data_path = '../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json'
df=pd.read_json(data_path,lines=True)

print(df.head())
sentences = df['headline'].to_list()

labels = df['is_sarcastic'].to_list()

urls = df['article_link'].to_list()



print(sentences)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(oov_token="<OOV>")



tokenizer.fit_on_texts(sentences)



word_index = tokenizer.word_index



sequences = tokenizer.texts_to_sequences(sentences)



print(sequences)

from tensorflow.keras.preprocessing.sequence import pad_sequences



padded_seq = pad_sequences(sequences)



print(padded_seq)
vocab_size = 10000

embedding_dim = 16

max_length = 21

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>'

training_size = 20000
training_sentences = np.array(sentences[0:training_size])

testing_sentences = np.array(sentences[training_size:])

training_labels = np.array(labels[0:training_size])

testing_labels = np.array(labels[training_size:])
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)



word_index = tokenizer.word_index



training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)



testing_sequence = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequence,maxlen=max_length,padding=padding_type,truncating=trunc_type)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(21,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')  

])
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



num_epochs = 10
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded,testing_labels),verbose=2)
import matplotlib.pyplot as plt



def plot_graphs(history, value):

    plt.plot(history.history[value])

    plt.plot(history.history['val_'+value])

    plt.xlabel('Epochs')

    plt.ylabel(value)

    plt.legend([value,'val_'+value])

    plt.show()



plot_graphs(history,'accuracy')

plot_graphs(history,'loss')