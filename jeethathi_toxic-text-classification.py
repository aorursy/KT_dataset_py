# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

__print__ = print

def print(string):

    os.system(f'echo \"{string}\"')

    __print__(string)
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv')

print('Imported train and test')
train.head()
test.head()
train.isnull().any()
test.isnull().any()
x_train = train['comment_text']

y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

x_test = test['comment_text']
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer()

tokenizer.fit_on_texts(x_train)

print('Fit tokenizer on texts')
x_tokenized_train = tokenizer.texts_to_sequences(x_train)

x_tokenized_test = tokenizer.texts_to_sequences(x_test)

print('Converted x_train and x_test to tokenized form')
lengths = [len(comment) for comment in x_tokenized_train]

print(f'The longest comment is {max(lengths)} words long.')

sns.distplot(lengths, kde=False)
from keras.preprocessing.sequence import pad_sequences



max_length = 200

X_train = pad_sequences(x_tokenized_train, maxlen=max_length)

X_test = pad_sequences(x_tokenized_test, maxlen=max_length)
len(tokenizer.word_index)
from keras.models import Sequential

from keras.layers import Embedding, LSTM, GlobalAveragePooling1D, Dropout, Dense, LeakyReLU, Activation, GlobalMaxPool1D

from keras import metrics



num_features, embed_size = len(tokenizer.word_index), 128

metric = ['accuracy']



models = []



model1 = Sequential()

model1.add(Embedding(num_features + 1, embed_size, input_length=max_length))

model1.add(LSTM(64, return_sequences=True))

model1.add(GlobalAveragePooling1D())

model1.add(Dropout(0.1))

model1.add(Dense(32))

model1.add(Activation('relu'))

model1.add(Dropout(0.1))

model1.add(Dense(16))

model1.add(Activation('relu'))

model1.add(Dropout(0.1))

model1.add(Dense(6, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



models.append(model1)



model2 = Sequential()

model2.add(Embedding(num_features + 1, embed_size, input_length=max_length))

model2.add(LSTM(64, return_sequences=True))

model2.add(GlobalMaxPool1D())

model2.add(Dropout(0.1))

model2.add(Dense(32))

model2.add(Activation('relu'))

model2.add(Dropout(0.1))

model2.add(Dense(16))

model2.add(Activation('relu'))

model2.add(Dropout(0.1))

model2.add(Dense(6, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



models.append(model2)



model3 = Sequential()

model3.add(Embedding(num_features + 1, embed_size, input_length=max_length))

model3.add(LSTM(64, return_sequences=True))

model3.add(GlobalMaxPool1D())

model3.add(Dropout(0.05))

model3.add(Dense(6, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



models.append(model3)



model4 = Sequential()

model4.add(Embedding(num_features + 1, embed_size, input_length=max_length))

model4.add(LSTM(64, return_sequences=True))

model4.add(GlobalMaxPool1D())

model4.add(Dropout(0.1))

model4.add(Dense(32, activation='relu'))

model4.add(Dropout(0.05))

model4.add(Dense(6, activation='sigmoid'))

model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



models.append(model4)



del model1, model2, model3, model4



# # 0.958

# models[0].add(Embedding(num_features + 1, embed_size, input_length=max_length))

# models[0].add(LSTM(64, return_sequences=True))

# models[0].add(GlobalAveragePooling1D())

# models[0].add(Dropout(0.1))

# models[0].add(Dense(48))

# models[0].add(LeakyReLU())

# models[0].add(Dropout(0.1))

# models[0].add(Dense(6, activation='sigmoid'))

# models[0].compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



# # 0.961

# models[1].add(Embedding(num_features + 1, embed_size, input_length=max_length))

# models[1].add(LSTM(64, return_sequences=True))

# models[1].add(GlobalAveragePooling1D())

# models[1].add(Dropout(0.1))

# models[1].add(Dense(48))

# models[1].add(Activation('relu'))

# models[1].add(Dropout(0.1))

# models[1].add(Dense(6, activation='sigmoid'))

# models[1].compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



# # 0.962

# models[2].add(Embedding(num_features + 1, embed_size, input_length=max_length))

# models[2].add(LSTM(64, return_sequences=True))

# models[2].add(GlobalAveragePooling1D())

# models[2].add(Dropout(0.1))

# models[2].add(Dense(32))

# models[2].add(Activation('relu'))

# models[2].add(Dropout(0.1))

# models[2].add(Dense(16))

# models[2].add(Activation('relu'))

# models[2].add(Dropout(0.1))

# models[2].add(Dense(6, activation='sigmoid'))

# models[2].compile(loss='binary_crossentropy', optimizer='adam', metrics=metric)



print('Created models')

models
import time



batch_size = 16

validation_split = 0.1

epochs = 3



histories = []



for i, model in enumerate(models):

    print(f'Beginning to fit model {i}')

    start_time = time.time()

    history = model.fit(X_train, y_train,

                        validation_split=validation_split,

                        batch_size=batch_size,

                        epochs=epochs)

    histories.append(history)

    end_time = time.time()

    print(f'Fit model {i} in {end_time - start_time} seconds.')
from IPython.display import display

for i, history in enumerate(histories):

    print(f'Model {i}')

    h = pd.DataFrame(history.history)

    h.index.name = 'epoch'

    display(h)
y_preds = []



for i, model in enumerate(models):

    print(f'Started predicting for model {i}')

    y_pred = model.predict(X_test, batch_size=4096)

    y_preds.append(y_pred)

    print(f'Predicted stuff for model {i}')
y = []

for i, model in enumerate(models):

    y_i = pd.DataFrame(data=y_preds[i],

                        columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

    y_i = pd.concat([test['id'], y_i], axis=1)

    y.append(y_i)

# y = pd.DataFrame(data=y_pred, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

# y = pd.concat([test['id'], y], axis=1)

y
for i, y_i in enumerate(y):

    filename = f'submision_{i}.csv'

    y_i.to_csv(filename, index=False)

    print(f'Created file {filename}')

# y.to_csv('submission.csv', index=False)