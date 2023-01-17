# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the import libraries

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam
#creating a corpus of words by reading in the data

tokenizer = Tokenizer()

data = open('/kaggle/input/text-generation/donald_trump.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)

print(total_words)
input_sequences = []

for line in corpus:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i+1]

        input_sequences.append(n_gram_sequence)



# pad sequences 

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))



# create predictors and label

xs, labels = input_sequences[:,:-1],input_sequences[:,-1]



ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
print(tokenizer.word_index)
model = Sequential()

model.add(Embedding(total_words, 128, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(256)))

model.add(Dense(total_words, activation='softmax'))

adam = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit(xs, ys, epochs=12, verbose=1)

#print model.summary()

print(model)
import matplotlib.pyplot as plt

def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.xlabel("Epochs")

    plt.ylabel([string])

    plt.show()
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
seed_text = "As we restore American leadership throughout the world, we are once again standing up for freedom in our hemisphere."

next_words = 50

  

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    predicted = model.predict_classes(token_list, verbose=0)

    output_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted:

            output_word = word

            break

    seed_text += " " + output_word

print(seed_text)