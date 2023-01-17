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

from keras.callbacks import ModelCheckpoint, EarlyStopping
clickbait_data = pd.read_csv('/kaggle/input/clickbait-dataset/clickbait_data.csv')

clickbait_data.head()
h_d = []

for i in clickbait_data['headline']:

    h_d.append(i.split())

print(h_d[:2])
w2vc_model = Word2Vec(h_d, size=50, workers=32, min_count=1, window=3)

print(w2vc_model)
# tokenize the data



token = Tokenizer(35789)

token.fit_on_texts(clickbait_data['headline'])

text = token.texts_to_sequences(clickbait_data['headline'])

text = pad_sequences(text)
y = clickbait_data['clickbait'].values
# split the data into train test split



X_train, X_test, y_train, y_test = train_test_split(np.array(text), y, test_size=0.2,stratify=y)
# build the model



model = Sequential()

model.add(w2vc_model.wv.get_keras_embedding(True))

model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.2))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()
# compile and train model



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=3)
model.save('clickbaitmodel2')
preds = [round(i[0]) for i in model.predict(X_test)]

cm = confusion_matrix(y_test, preds)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.xticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)

plt.yticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)

plt.show()
test = ['Which TV Female Friend Group Do You Belong In', 'What The Most Beautiful College Campus In The World', 

        'A tour of Japan\'s Kansai region', 'These BFFs Are Slaying Internet Fashion']

token_text = pad_sequences(token.texts_to_sequences(test))

preds = [round(i[0]) for i in model.predict(token_text)]

for (text, pred) in zip(test, preds):

    label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'

    print("{} - {}".format(text, label))