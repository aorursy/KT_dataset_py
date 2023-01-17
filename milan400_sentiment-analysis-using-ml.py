import numpy as np 

import pandas as pd 

import random

import re
test_data =  pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
data = {}

for a, b in zip(test_data['text'], test_data['sentiment']):

    data[a] = b
def seperate(data):

    X = []

    Y = []

    

    for K, V in data.items():

        K = K.lower()

        

        X.append(K)

        Y.append(V)

    return X,Y
X, Y = seperate(data)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model = Pipeline([

     ('vect', CountVectorizer()),

     ('tfidf', TfidfTransformer()),

     ('clf', MultinomialNB()),

])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.80)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, X_test, y_test)

plt.show()