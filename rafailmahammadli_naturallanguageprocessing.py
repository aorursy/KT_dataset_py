import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

dataset
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):

  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

  review = review.lower()

  review = review.split()

  ps = PorterStemmer()

  all_stopwords = stopwords.words('english')

  all_stopwords.remove('not')

  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

  review = ' '.join(review)

  corpus.append(review)
print(corpus)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, -1].values

len(X[0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 0)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)

new_review = 'I was disgusted because I was pretty sure that was human hair.'

new_review = re.sub('[^a-zA-Z]', ' ', new_review)

new_review = new_review.lower()

new_review = new_review.split()

ps = PorterStemmer()

all_stopwords = stopwords.words('english')

all_stopwords.remove('not')

new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]

new_review = ' '.join(new_review)

new_corpus = [new_review]

new_X_test = cv.transform(new_corpus).toarray()

new_y_pred = classifier.predict(new_X_test)

print(new_y_pred)