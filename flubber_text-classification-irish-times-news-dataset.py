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
irishtimes_data = pd.read_csv('../input/ireland-historical-news/irishtimes-date-text.csv')

irishtimes_data.shape
irishtimes_data.head(10)
irishtimes_data.dtypes
irishtimes_data.headline_category = irishtimes_data.headline_category.astype('category')
irishtimes_data.headline_category.cat.categories
# Remove blank rows if any

irishtimes_data.headline_text.dropna(inplace=True)

# Change all the text to lower case

irishtimes_data.headline_text = [entry.lower() for entry in irishtimes_data.headline_text]

# Replace punctuation symbols by space

import re

PUNCTUATIONS = re.compile('[/(){}\[\]\|@,;\']')

irishtimes_data.headline_text = [PUNCTUATIONS.sub('', entry) for entry in irishtimes_data.headline_text]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(irishtimes_data.headline_text, irishtimes_data.headline_category, test_size=0.2, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
import nltk

from sklearn.feature_extraction.text import CountVectorizer

#nltk.download()

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)



class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
#count_vect = CountVectorizer(stop_words='english')

count_vect = StemmedCountVectorizer(stop_words='english')

count_vect.fit(X_train)

X_train_counts = count_vect.transform(X_train)

X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB().fit(X_train_tfidf, y_train)
X_test_counts = count_vect.transform(X_test)

X_test_counts.shape
tfidf_transformer = TfidfTransformer()

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

X_test_tfidf.shape
predicted = nb.predict(X_train_tfidf)

np.mean(predicted == y_train)
predicted = nb.predict(X_test_tfidf)

from sklearn import metrics

metrics.accuracy_score(y_test, predicted)
w3_latnigrin_data = pd.read_csv('../input/ireland-historical-news/w3-latnigrin-text.csv')

#count_vect = CountVectorizer(vocabulary=count_vect.vocabulary)

X_new_data_counts = count_vect.transform(w3_latnigrin_data.headline_text)

X_new_data_counts.shape
tfidf_transformer = TfidfTransformer()

X_new_data_tfidf = tfidf_transformer.fit_transform(X_new_data_counts)

X_new_data_tfidf.shape
predicted = nb.predict(X_new_data_tfidf)
w3_latnigrin_data['headline_category_NB'] = predicted

w3_latnigrin_data.head()
from sklearn.linear_model import SGDClassifier

svm = SGDClassifier(loss='hinge',

    penalty='l2',

    alpha=0.0001,

    l1_ratio=0.15,

    fit_intercept=True,

    max_iter=1000,

    tol=0.001,

    shuffle=True,

    verbose=0,

    epsilon=0.1,

    n_jobs=None,

    random_state=7,

    learning_rate='optimal',

    eta0=0.0,

    power_t=0.5,

    early_stopping=False,

    validation_fraction=0.1,

    n_iter_no_change=5,

    class_weight=None,

    warm_start=False,

    average=False,).fit(X_train_tfidf, y_train)
predicted = svm.predict(X_train_tfidf)

np.mean(predicted == y_train)
predicted = svm.predict(X_test_tfidf)

metrics.accuracy_score(y_test, predicted)
predicted = svm.predict(X_new_data_tfidf)

w3_latnigrin_data['headline_category_SVM'] = predicted

w3_latnigrin_data.head()
from keras.preprocessing.text import Tokenizer

# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each text.

MAX_SEQUENCE_LENGTH = 250

# This is fixed.

EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(irishtimes_data.headline_text)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
from tflearn.data_utils import pad_sequences

X = tokenizer.texts_to_sequences(irishtimes_data.headline_text)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(irishtimes_data.headline_category).values

print('Shape of label tensor:', Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Embedding

from keras.layers import SpatialDropout1D

from keras.layers import LSTM

from keras.callbacks import EarlyStopping



model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(156, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



epochs = 1

batch_size = 64



history = model.fit(X_train, Y_train, 

                    epochs=epochs, 

                    batch_size=batch_size,

                    validation_split=0.1,

                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]

                   )
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
seq = tokenizer.texts_to_sequences(w3_latnigrin_data.headline_text)

padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

pred = model.predict(padded)



labels = irishtimes_data.headline_category.cat.categories

print(pred, labels[np.argmax(pred)])
w3_latnigrin_data['headline_category_LSTM'] = labels[np.argmax(pred)]

w3_latnigrin_data.head()