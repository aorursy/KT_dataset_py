import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords



from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding



from keras.layers import Dense,LSTM

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
train = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")

test = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")
train.head()
test.head()
print (train["Phrase"][0])

print (train["Phrase"][1])

print (train.shape)




import seaborn as sns



sns.countplot(x='Sentiment', data=train)


tokenizer = Tokenizer()



full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

tokenizer.fit_on_texts(full_text)



X_train = tokenizer.texts_to_sequences(train['Phrase'].values)

X_test = tokenizer.texts_to_sequences(test['Phrase'].values)
max_len = 50

X_train = pad_sequences(X_train, maxlen = max_len)

X_test = pad_sequences(X_test, maxlen = max_len)

len(X_train)
X_train = np.array(X_train)

X_test = np.array(X_test)
X_train.shape
y = train['Sentiment']
vocab_size = len(tokenizer.word_index) + 1

vocab_size
# TEST

vocabulary_size = len(tokenizer.word_counts)

vocabulary_size = vocabulary_size + 1

seq_len = X_train.shape[1]

model_test = Sequential()

model_test.add(Embedding(input_dim = vocabulary_size, output_dim = 2, input_length=seq_len))

model_test.compile('rmsprop', 'mse')

output_array = model_test.predict(X_train)

print (output_array.shape)

out1 = pd.DataFrame(output_array[0])

out1.tail()
vocabulary_size = len(tokenizer.word_counts)

vocabulary_size = vocabulary_size + 1

seq_len = X_train.shape[1]



model = Sequential()

model.add(Embedding(vocabulary_size, 25, input_length=seq_len))

model.add(LSTM(150, return_sequences=True)) # to stack LSTM we need return seq 

model.add(LSTM(150))

model.add(Dense(150, activation='relu'))



model.add(Dense(5, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
from keras.utils import to_categorical

y = to_categorical(y, num_classes=5)



y.shape
# fit model

model.fit(X_train, y, batch_size=256, epochs=10,verbose=1)
sub = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')

sub.head()
pred = model.predict(X_test, batch_size = 256, verbose = 1)

pred[0]
import matplotlib.pyplot as plt

predictions = np.round(np.argmax(pred, axis=1)).astype(int)

plt.hist(predictions, normed=False, bins=5)
sub['Sentiment'] = predictions

sub.to_csv("submission.csv", index=False)