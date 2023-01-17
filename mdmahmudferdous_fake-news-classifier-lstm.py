import numpy as np

import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/fake-news/train.csv')

test=pd.read_csv('/kaggle/input/fake-news/test.csv')

submit=pd.read_csv('/kaggle/input/fake-news/submit.csv')
train.head()
df=train.dropna()
X=df['title']

y=df['label']
import tensorflow as tf

tf.__version__
from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.preprocessing.sequence import pad_sequences
voc_size=5000
X=[i.lower() for i in X]
onehot=[one_hot(words,voc_size) for words in X]
sen_len=30

embedded_doc=pad_sequences(onehot, padding='pre', maxlen=sen_len)

print(embedded_doc)
embedding_vector_feature=40

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_feature, input_length=sen_len))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
X_final=np.array(embedded_doc)

y_final=np.array(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.33, random_state=0)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)