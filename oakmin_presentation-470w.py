# ALL imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style; style.use('ggplot')

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier
# Create dataframes train and test

train = pd.read_csv('../input/drugsComTrain_raw.csv')

test = pd.read_csv('../input/drugsComTest_raw.csv')
train.head()
b = "'@#$%^()&*;!.-"

X_train = np.array(train['review'])

X_test = np.array(test['review'])



def clean(X):

    for index, review in enumerate(X):

        for char in b:

            X[index] = X[index].replace(char, "")

    return(X)



X_train = clean(X_train)

X_test = clean(X_test)

print(X_train[:2])
from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from keras.utils import to_categorical

from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer

import nltk



vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),lowercase=True, max_features=5000)

#vectorizer = TfidfVectorizer(binary=True, stop_words=stopwords.words('english'), lowercase=True, max_features=5000)

test_train = np.concatenate([X_train, X_test])

print(test_train.shape)

X_onehot = vectorizer.fit_transform(test_train)

stop_words = vectorizer.get_stop_words()

print(type(X_onehot))
print(X_onehot.shape)

print(X_onehot.toarray())
names_list = vectorizer.get_feature_names()

names = [[i] for i in names_list]

names = Word2Vec(names, min_count=1)

print(len(list(names.wv.vocab)))

print(list(names.wv.vocab)[:5])

def score_transform(X):

    y_reshaped = np.reshape(X['rating'].values, (-1, 1))

    for index, val in enumerate(y_reshaped):

        if val >= 8:

            y_reshaped[index] = 1

        elif val >= 5:

            y_reshaped[index] = 2

        else:

            y_reshaped[index] = 0

    y_result = to_categorical(y_reshaped)

    return y_result

    

    print(X_onehot)
y_train_test = pd.concat([train, test], ignore_index=True)

y_train = score_transform(y_train_test)

print(y_train)

print(y_train.shape)
len(vectorizer.get_feature_names())
from numpy.random import seed



np.random.seed(1)

model = Sequential()

model.add(Dense(units=256, activation='relu', input_dim=len(vectorizer.get_feature_names())))

model.add(Dense(units=3, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary
history = model.fit(X_onehot[:-53866], y_train[:-53866], epochs=6, batch_size=128, verbose=1, validation_data=(X_onehot[157382:], y_train[157382:]))
scores = model.evaluate(X_onehot[157682:], y_train[157682:], verbose=1)
scores[1]
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()