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
data = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
data
data.airline_sentiment.value_counts().plot(kind = "bar")
data.airline_sentiment_confidence.hist()
data = data[data.airline_sentiment_confidence > .5]

data = data[data.airline_sentiment != "neutral"]
data
from nltk.tokenize import TweetTokenizer

from nltk.corpus import  stopwords

import string

sw = stopwords.words("english")

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

def clean(tweet):

    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).lower()

    words = [word for word in tokenizer.tokenize(tweet) if word not in set(sw)]

    return " ".join(words)

data["clean_text"] = data.text.apply(clean)

data
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()

X = cv.fit_transform(data.clean_text)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

Y = encoder.fit_transform(data.airline_sentiment.values.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')

lr.fit(x_train, y_train)
from sklearn.metrics import classification_report, accuracy_score
y_pred = lr.predict(x_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data.clean_text)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2)
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')

lr.fit(x_train, y_train)
from sklearn.metrics import classification_report, accuracy_score

y_pred = lr.predict(x_test)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
cv = CountVectorizer()

X = cv.fit_transform(data.clean_text)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train.toarray(), y_train)

y_pred = gnb.predict(x_test.toarray())

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

sgd.fit(x_train.toarray(), y_train)

y_pred = sgd.predict(x_test.toarray())

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras import regularizers
max_fatures = 5000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data.clean_text.values)

X = tokenizer.texts_to_sequences(data.clean_text.values)

X = pad_sequences(X)
embed_dim = 512

lstm_out = 64



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(SpatialDropout1D(0.5))

#model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3,  return_sequences=True))

#model.add(SpatialDropout1D(0.5))

model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.5))

model.add(Dense(64,activation='tanh', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
Y = pd.get_dummies(data.airline_sentiment).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)

batch_size = 64

model.fit(X_train, Y_train, epochs = 3, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))