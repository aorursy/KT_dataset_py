# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
def prepare_data(data):

    # Keeping only the neccessary columns

    data = data[['text','target']]

    data['text'] = data['text'].apply(lambda x: x.lower())

    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))



    print(data[ data['target'] == 0].size)

    print(data[ data['target'] == 1].size)

    for idx,row in data.iterrows():

        row[0] = row[0].replace('rt',' ')



    max_fatures = 2000

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')

    tokenizer.fit_on_texts(data['text'].values)

    X = tokenizer.texts_to_sequences(data['text'].values)

    X = pad_sequences(X)

    return X
def prepare_test_data(data):

    # Keeping only the neccessary columns

    data = data[['text']]

    data['text'] = data['text'].apply(lambda x: x.lower())

    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    for idx,row in data.iterrows():

        row[0] = row[0].replace('rt',' ')



    max_fatures = 2000

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')

    tokenizer.fit_on_texts(data['text'].values)

    X = tokenizer.texts_to_sequences(data['text'].values)

    X = pad_sequences(X,maxlen=30)

    return X
X = prepare_data(df)
embed_dim = 128

lstm_out = 196

max_fatures = 2000

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
Y = pd.get_dummies(df['target']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
batch_size = 32

model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)
validation_size = 1500



X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test
X_test = prepare_test_data(test)
X_test.shape
preds=model.predict_classes(X_test)
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission["target"]=list(preds)
submission.to_csv("submission.csv",index=False)
submission