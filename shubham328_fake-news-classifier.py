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
train=pd.read_csv("../input/fake-news/train.csv")
test=pd.read_csv("../input/fake-news/test.csv")
train
test
train=train.dropna()
test=test.dropna()
train.reset_index(inplace=True)
test.reset_index(inplace=True)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet=WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
corpus=[]
for i in range(len(train)):
    review = re.sub('[^a-zA-Z]', ' ', train.title[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# for i in range(len(test)):
#     review = re.sub('[^a-zA-Z]', ' ', test.title[i])
#     review = review.lower()
#     review = review.split()
#     review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus.append(review)
corpus
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
Y=train.label
Y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y)
from sklearn.naive_bayes import MultinomialNB
Fake_news_model= MultinomialNB().fit(X_train,y_train)
y_pred = Fake_news_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(X_train, y_train)
y_pred = linear_clf.predict(X_test)
accuracy_score(y_test,y_pred)
