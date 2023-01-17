import pandas as pd

import numpy as np

import re

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

import nltk

from nltk import word_tokenize

from string import punctuation

import os

print(os.listdir("../input"))
cols = ['sentiment','id','date','query_string','user','text']

df = pd.read_csv('../input/training.1600000.processed.noemoticon.csv', encoding="latin1", names=cols)
df.head()
np.random.seed(0)

index = np.random.randint(low=0, high=1599999, size=10000)

df2 = df.loc[index, ['sentiment', 'text']].reset_index(drop=True)

df2['sentiment'] = df2['sentiment'].replace({4:1})
df2.head()
df2['sentiment'].value_counts()
pat1 = '@[^ ]+'

pat2 = 'http[^ ]+'

pat3 = 'www.[^ ]+'

pat4 = '#[^ ]+'

pat5 = '[0-9]'

pat6 = '[.]'

pat7 = '[-]'

pat8 = '[*]'

pat9 = '[_]'

combined_pat = '|'.join((pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8, pat9))
clean_tweet_texts = []

for t in df2['text']:

    t = t.lower()

    stripped = re.sub(combined_pat, '', t)

    negations = re.sub("n't", " not", stripped)

    negations = negations.replace('-', ' ')

    negations = negations.replace('_', ' ')

    clean_tweet_texts.append(negations)
clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])

clean_df ['sentiment'] = df2['sentiment']
x = clean_df['text']

y = clean_df['sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
pos_df = clean_df.loc[clean_df['sentiment'] == 1].reset_index(drop=True)
neg = clean_df.loc[clean_df['sentiment'] == 0].reset_index(drop=True)
pos_df.head()
stop = set(stopwords.words('english')) 
stop.remove('not')
stop
cv = CountVectorizer(stop_words=stop, binary=False, ngram_range=(1,3))
cv.fit(x_train)
cv.get_feature_names()
len(cv.get_feature_names())
x_train_cv = cv.transform(x_train)

x_test_cv = cv.transform(x_test)
x_train_cv.toarray()
#Apply tfidf vectorizer

tv = TfidfVectorizer(stop_words='english', binary=False, ngram_range=(1,3))



tv.fit(x_train)



tv.get_feature_names()



len(tv.get_feature_names())



tv.vocabulary_



x_train_tv = tv.transform(x_train)

x_test_tv = tv.transform(x_test)



x_train_tv



x_train_tv.toarray()
x_train_tv.shape
from keras import regularizers

from keras.layers import LeakyReLU
def deepModel(shape):

    model = Sequential()

    model.add(Dense(units = 1024, activation = 'relu', input_dim = shape))

    model.add(Dense(256))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dropout(0.1))

    model.add(Dense(128))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dense(64))

    model.add(Dropout(0.6))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
def showHist(trained):

    plt.plot(trained.history['acc'])

    plt.plot(trained.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(trained.history['loss'])

    plt.plot(trained.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
model1 = deepModel(x_train_tv.shape[1])

model1.summary()

hist1 = model1.fit(x_train_tv, y_train, epochs=5, batch_size=32, validation_data = (x_test_tv, y_test))
showHist(hist1)
model2 = deepModel(x_train_cv.shape[1])

hist2 = model2.fit(x_train_cv, y_train, epochs=5, batch_size=32, validation_data = (x_test_cv, y_test))
showHist(hist2)
def wideModel(shape):

    model = Sequential()

    model.add(Dense(units = 1024, activation = 'relu', input_dim = shape))

    model.add(Dense(1024))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model3 = wideModel(x_train_cv.shape[1])

hist3 = model3.fit(x_train_cv, y_train, epochs=5, batch_size=32, validation_data = (x_test_cv, y_test))
showHist(hist3)
model4 = wideModel(x_train_tv.shape[1])

hist4 = model4.fit(x_train_tv, y_train, epochs=5, batch_size=32, validation_data = (x_test_tv, y_test))
showHist(hist4)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
lr1 = LogisticRegression()

lr1.fit(x_train_cv, y_train)

pred1 = lr1.predict(x_test_cv)

print(accuracy_score(y_test, pred1))
lr2 = LogisticRegression()

lr2.fit(x_train_tv, y_train)

pred2 = lr2.predict(x_test_tv)

print(accuracy_score(y_test, pred2))