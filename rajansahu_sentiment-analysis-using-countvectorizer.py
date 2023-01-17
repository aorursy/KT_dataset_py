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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("/kaggle/input/imdb-sentiments/train.csv")
df_train.head
import re
corpus = []
for i in range(0, 25000):
    review = re.sub(r'\W', ' ', str(df_train["text"][i]))
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)
len(corpus)
# Creating the BOW model
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 25000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
count_vect = vectorizer.fit_transform(corpus).toarray()
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
train, test, train_y, test_y = train_test_split(count_vect, df_train['sentiment'], test_size = 0.20, random_state = 0)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
# Training the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train,train_y)
# Testing model performance
sent_pred = classifier.predict(test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, sent_pred)
print(cm)
# Naive Bayes on Count Vectors
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn import decomposition, ensemble
classifier2= naive_bayes.MultinomialNB()
classifier2.fit(train,train_y)
# Testing model performance
sent_pred = classifier2.predict(test)
cm = confusion_matrix(test_y, sent_pred)
print(cm)
classifier3=ensemble.RandomForestClassifier()
classifier3.fit(train,train_y)
sent_pred = classifier3.predict(test)
cm = confusion_matrix(test_y, sent_pred)
print(cm)

