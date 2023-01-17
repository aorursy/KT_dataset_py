import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/winemag-data_first150k.csv')

X = dataset.iloc[:, 2].values

y = dataset.iloc[:, 4].values

# Printing out the first 5 values of the array

%matplotlib inline

dataset.head()
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import re



lem = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

prog = re.compile(r'[^a-zA-Z0-9]')

filtered = []



for i in range(len(X)):

    tokenized_sentence = word_tokenize(X[i])

    filtered_sentence = []

    for w in tokenized_sentence:

        w = w.lower()

        w = lem.lemmatize(w, "v")

        if not prog.match(w):

            if w not in stop_words:

                filtered_sentence.append(w)

    filtered.append(filtered_sentence)



print(filtered[:5])
Review_count = dataset.groupby('points').count()

plt.bar(Review_count.index.values, Review_count['description'])

plt.xlabel('Review Points')

plt.ylabel('Number of reviews')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    text_counts, y, test_size=0.3, random_state=0)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, mean_squared_error  

clf = MultinomialNB().fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("MultinomialNB Accuracy: ", accuracy_score(y_test, y_pred))

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

text_tf = tf.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_tf, y, test_size=0.3, random_state=0)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, mean_squared_error  

clf = MultinomialNB().fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("MultinomialNB Accuracy: ", accuracy_score(y_test, y_pred))

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))