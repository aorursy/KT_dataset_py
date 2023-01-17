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
import numpy as np

import pandas as pd

import csv

import tensorflow as tf

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D

from sklearn.model_selection import train_test_split

print(tf.__version__)
train_initial = pd.read_csv('../input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv').fillna(' ')

#print(train_df.count)

valid_df = pd.read_csv('../input/imdb-dataset-sentiment-analysis-in-csv-format/Valid.csv').fillna(' ')

#print(valid_df.count)



train_df = train_initial.append(valid_df, ignore_index = True) 

#print(train_df.count)

train_df.sample(10, random_state=1)
x = train_df['text'].values

print(x)
y = train_df['label'].values
train_df['label'].plot(kind='hist', title='Distribution')
train_df['label'].value_counts()
max_features = 10000

max_text_length = 400



x_tokenizer = text.Tokenizer(max_features)

x_tokenizer.fit_on_texts(list(x))

x_tokenized = x_tokenizer.texts_to_sequences(x)

x_train_val = sequence.pad_sequences(x_tokenized, maxlen=max_text_length)
embedding_dim = 100

embeddings_index = dict()

f = open('../input/glove6b100dtxt/glove.6B.100d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print(f'Found {len(embeddings_index)} word vectors')
embedding_matrix = np.zeros((max_features, embedding_dim))

for word, index in x_tokenizer.word_index.items():

    if index > max_features -1:

        break

    else:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector
model = Sequential()

model.add(Embedding(max_features,

                   embedding_dim,

                   embeddings_initializer=tf.keras.initializers.Constant(

                   embedding_matrix),

                   trainable=False))

model.add(Dropout(0.2))
filters = 250

kernel_size = 3

hidden_dims = 250
model.add(Conv1D(filters,

                kernel_size,

                padding='valid'))

model.add(MaxPooling1D())

model.add(Conv1D(filters,

         5,

         padding='valid',

         activation='relu'))

model.add(MaxPooling1D())

model.add(Conv1D(filters,

         5,

         padding='valid',

         activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y,

                                                 test_size=0.15, random_state=1)
batch_size = 32

epochs = 15



model.fit(x_train, y_train,

         batch_size=batch_size,

         epochs=epochs,

         validation_data=(x_val, y_val))
test_df = pd.read_csv('../input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv').fillna(' ')
x_test = test_df['text'].values

y_test = test_df['label'].values
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)

x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)
y_testing = model.predict(x_testing, verbose=1, batch_size=32)
model.evaluate(x_testing, y_test, batch_size=32)