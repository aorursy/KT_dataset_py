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
dataset = pd.read_csv('../input/nlp-getting-started/train.csv')
dataset.head()
dataset['text'].isnull().sum()
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#We need to vectorize the text to provide it as input to the model

max_features = 30000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(dataset['text'].values)
X_train = tokenizer.texts_to_sequences(dataset['text'].values)
X_train = pad_sequences(X_train)
Y_train = dataset.iloc[:, -1].values
#X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X_train.shape,Y_train.shape)

import tensorflow as tf
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

def get_model():
    embed_dim = 150
    lstm_out = 200
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_out))
    model.add(Dropout(0.25))
    model.add(Dense(2,activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    
    return model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn= get_model, epochs=10, batch_size=32, verbose=0 )
model.fit(X_train, Y_train)
test = pd.read_csv('../input/nlp-getting-started/test.csv')
X_test = tokenizer.texts_to_sequences(test['text'].values)
X_test = pad_sequences(X_test)
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

results = cross_val_score(model, X_train, Y_train, cv=5, scoring="f1")
print(results)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission['target'] = model.predict(X_test)
sample_submission.head(10)
sample_submission.to_csv('submission1.csv', index=False)