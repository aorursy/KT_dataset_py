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
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from sklearn import preprocessing



from gensim.models import Word2Vec



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, MaxPool1D, Dropout, Dense, GlobalMaxPooling1D, Embedding, Activation

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
spam_ham_data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', delimiter=',', encoding='latin-1')

spam_ham_data.head()
spam_ham_data = spam_ham_data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

spam_ham_data = spam_ham_data.rename(columns={'v1':'label', 'v2':'text'})

spam_ham_data.head()
# list of lists of text for word2vec

sh_d = []

for i in spam_ham_data['text']:

    sh_d.append(i.split())

print(sh_d[:2])
w2vsh_model = Word2Vec(sh_d, size=50, workers=32, min_count=1, window=3)

print(w2vsh_model)
# tokenize the data



token = Tokenizer(15585)

token.fit_on_texts(spam_ham_data['text'])

text = token.texts_to_sequences(spam_ham_data['text'])

text = pad_sequences(text)
label = preprocessing.LabelEncoder()

y = label.fit_transform(spam_ham_data['label'])

y = to_categorical(y)

print(y[:2])
# split the data into train test split



X_train, X_test, y_train, y_test = train_test_split(np.array(text), y, test_size=0.2,stratify=y)
# build the model



model = Sequential()

model.add(w2vsh_model.wv.get_keras_embedding(True))

model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.2))

model.add(Dense(2))

model.add(Activation('sigmoid'))

model.summary()
# compile and train model



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=5)
# print labels



labels = label.classes_

print(labels)
# check prediction



predicted = model.predict(X_test)
for i in range(10,50,3):

    print(spam_ham_data['text'].iloc[i][:50], "...")

    print("Actual category: ", labels[np.argmax(y_test[i])])

    print("predicted category: ", labels[np.argmax(predicted[i])])