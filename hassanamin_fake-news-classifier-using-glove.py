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
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



df = pd.read_csv('../input/textdb3/fake_or_real_news.csv') # Load data into DataFrame

# Pre-Processing

df['text'] = df['text'].apply(lambda x: x.lower())


#max_features = 2000 # Vocabulary Size



#tokenizer = Tokenizer(num_words=max_features, split=' ')

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df['text'].values)

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary Size :- ",vocab_size)

X = tokenizer.texts_to_sequences(df['text'].values)

max_length = 1000

# Padding

X = pad_sequences(X,maxlen = max_length, padding = 'post')
#y = df.label

y = pd.get_dummies(df['label']).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=53)



print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from numpy import array

from numpy import asarray

from numpy import zeros



# load the whole embedding into memory

embeddings_index = dict()

f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))



# Creating Embedding Matrix



# create a weight matrix for words in training docs

embedding_matrix = zeros((vocab_size, 200))

for word, i in tokenizer.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



# define the model

model = Sequential()

model.add(Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False))

model.add(Flatten())

model.add(Dense(2, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model

print(model.summary())

# fit the model

model.fit(X_train, y_train, epochs=50, verbose=0)



# evaluate the model

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: %f' % (accuracy*100))
