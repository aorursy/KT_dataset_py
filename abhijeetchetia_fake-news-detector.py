import numpy as np

import pandas as pd

import itertools

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
file = "../input/news-collections/news collections.csv"

data = pd.read_csv(file)
data.head(5)
data.isnull().any()
data['label'].value_counts()
labels = data['label']
x_train,x_test,y_train,y_test = train_test_split(data['text'], labels, test_size=0.2, random_state=5)
# Initialize a TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set

tfidf_train = tfidf_vectorizer.fit_transform(x_train) 

tfidf_test = tfidf_vectorizer.transform(x_test)
# Initializing model



model = PassiveAggressiveClassifier(C = 1, max_iter=100, random_state=10)

model.fit(tfidf_train, y_train)
# Generating predictions and calculating accuracy scores



y_pred = model.predict(tfidf_test)

score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {round(score * 100, 2)}%')
confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])