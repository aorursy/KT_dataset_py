# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#LSTM for sequence classification in the Imdb dataset.

import numpy as np

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

#Fix random seed for reproducibility

np.random.seed(6)
#Keras have built the vocablary of the words of the reviews and 

# sort the occurances of word in the review .

# Load the dataset but only keep the top n words, ignore the rest

top_words = 5000   # it means we will use our top 5000 words for Training only



old = np.load

np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)



np.load = old

del(old)



print(X_train[1])

print(type(X_train[1]))

print(len(X_train[1]))
# Truncate and/or pad input sequences

max_review_length = 600

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)



print(X_train.shape)

print(X_train[1])
#Create the model

embedding_vector_length = 32

model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train , nb_epoch=10, batch_size=64)

#Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" %(scores[1]*100))