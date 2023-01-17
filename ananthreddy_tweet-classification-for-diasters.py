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
import re

import math



import warnings

from tqdm import tqdm, trange



warnings.filterwarnings('ignore')
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print (train_data.shape)

train_data.head()
train_data['text'][0]
print ('No. of unique keywords: {}'.format(train_data['keyword'].nunique()))

print ('No. of unique locations: {}'.format(train_data['location'].nunique()))
import seaborn as sns



sns.countplot(train_data['target'])
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print (test_data.shape)

test_data.head()
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

tqdm.pandas()



def word_tokens(text):

    tokenized_text = text.split(' ')

    tokenized_text = [w for w in tokenized_text if not w in stop_words]

    clean_text = ",".join(tokenized_text)

    clean_text = clean_text.replace(',', ' ')

    return (clean_text)



train_data['clean_text'] = train_data['text'].progress_apply(lambda x: word_tokens(x))

test_data['clean_text'] = test_data['text'].progress_apply(lambda x: word_tokens(x))
train_data['clean_text'].head()
train_text = train_data['clean_text'].values
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_text)
train_data['tokenized_text'] = tokenizer.texts_to_sequences(train_data['clean_text'])

test_data['tokenized_text'] = tokenizer.texts_to_sequences(test_data['clean_text'])
train_data['tokenized_text'].head()
max_token_length = 0 

for i in trange(len(train_data)):

    token_length = len(train_data['tokenized_text'][i])

    if (token_length > max_token_length):

        max_token_length = token_length

        

print ('Max. Token Length : {}'.format(max_token_length))
max_token = 0

for i in trange(len(train_data)):

    token = np.max(train_data['tokenized_text'][i])

    if (token > max_token):

        max_token = token

        

print (max_token)
y_train = train_data[['target']]

X_train = train_data[['tokenized_text']]



X_test = test_data[['tokenized_text']]
from keras.preprocessing import sequence



max_token_length = 40 # allow it to be greater than the obtained max length

y_train = train_data[['target']]

X_train = sequence.pad_sequences(X_train['tokenized_text'], maxlen = max_token_length, padding = 'post')



X_test = sequence.pad_sequences(X_test['tokenized_text'], maxlen = max_token_length, padding = 'post')
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15)
print (X_train.shape)

print (y_train.shape)



print (X_val.shape)

print (y_val.shape)
from keras.models import Sequential

from keras.layers import Input, Embedding, Dense, Dropout, LSTM, Bidirectional

from keras.optimizers import Adam



model = Sequential()

model.add(Embedding(max_token+1, 512))

# model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.60))

model.add(LSTM(32))

model.add(Dropout(0.40))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 6, batch_size = 32)
test_prediction = model.predict(X_test)

test_prediction = np.argmax(test_prediction, axis = 1)
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print (sample_submission.shape)

sample_submission.head()
submission = pd.DataFrame({'id': test_data['id'], 'target': test_prediction})

submission.to_csv('submission.csv', index = False)