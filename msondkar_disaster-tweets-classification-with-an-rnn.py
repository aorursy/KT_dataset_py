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
from keras.models import Sequential, load_model

from keras.layers import LSTM, GRU

from keras.layers import Dense, Embedding, Bidirectional, Dropout, Flatten

from keras.optimizers import Adam, SGD

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
print('Training Dataset contain {} samples'.format(train.shape[0]))

print('Testing Dataset contain {} samples'.format(test.shape[0]))
train = train.drop(['id', 'keyword', 'location'], axis=1)

test = test.drop(['id', 'keyword', 'location'], axis=1)
y_train =  train['target'].values

X_train = train.drop(['target'], axis=1).values.reshape(len(train),)

X_test = test['text'].values.reshape(len(test),)
total_tweets = np.concatenate((X_train, X_test))

print('Total tweets : ', len(total_tweets))


tokenizer = Tokenizer()

tokenizer.fit_on_texts(total_tweets)



# Vocbvulary Size

vocab_size = len(tokenizer.word_index) + 1

print('Size of Vocabulary : ', vocab_size)
# Maximum length for padding sequence

maxlen = max(len(x.split()) for x in total_tweets)

print('Maximum length of tweet : ', maxlen)
X_train_token = tokenizer.texts_to_sequences(X_train)

X_test_token = tokenizer.texts_to_sequences(X_test)



print('Text before tokenized')

print(X_train[0])

print('\nText after tokenized')

print(X_train_token[0])
X_train_pad = pad_sequences(X_train_token, maxlen=maxlen, padding='post')

X_test_pad = pad_sequences(X_test_token, maxlen=maxlen, padding='post')



print('Tokenized text before padding')

print(X_train_token[0])

print('\nTokenized text after padding')

print(X_train_pad[0])
hidden_units = 128

embed_units = 100



model = Sequential()

model.add(Embedding(vocab_size, embed_units, input_length = maxlen))

model.add(Bidirectional(LSTM(hidden_units)))

model.add(Dropout(0.2))

#model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.summary()
learning_rate = 0.0001



model.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
batch_size = 512

num_itr = 5



model_history = model.fit(X_train_pad, y_train, 

                          batch_size=batch_size, 

                          epochs=num_itr, 

                          validation_split=0.2)
pred = model.predict(X_test_pad)
sub = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sub["target"] = pred

sub["target"] = sub["target"].apply(lambda x : 0 if x<=.5 else 1)
sub.to_csv("submit_1.csv", index=False)