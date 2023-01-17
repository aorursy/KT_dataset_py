import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data_train = pd.read_csv('../input/umich-si650-nlp/train.csv')

data_test = pd.read_csv('../input/umich-si650-nlp/test.csv')
data_train.head()
print(data_train.dtypes)

print(data_train.describe())

print(data_train.info())
x_tr=data_train['sentence']

y_tr=data_train['label']
data_train.label.value_counts()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x_tr = vectorizer.fit_transform(data_train.sentence)

x_ts = vectorizer.transform(data_test.sentence)
WordFrequency = pd.DataFrame({'Word': vectorizer.get_feature_names(), 'Count': x_tr.toarray().sum(axis=0)})
WordFrequency['Frequency'] = WordFrequency['Count'] / WordFrequency['Count'].sum()
plt.plot(WordFrequency.Frequency)

plt.xlabel('Word Index')

plt.ylabel('Word Frequency')

plt.show()
WordFrequency_sort = WordFrequency.sort_values(by='Frequency', ascending=False)

WordFrequency_sort.head()
from keras.models import Sequential

from keras.layers import Dense,Flatten, Dropout

from keras.layers.convolutional import Conv1D,MaxPooling1D

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split



embedding_dim = 32

vocab_size=200



model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=1903))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

#model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.50))

model.add(Flatten())

#model.add(Dense(250, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',

              metrics=['accuracy'])

model.summary()
X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.3)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model.fit(X_train, Y_train, batch_size=128, epochs = 10, validation_data = (X_test,Y_test))
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

print(test_acc*100)