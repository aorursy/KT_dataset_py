import pandas as pd

import numpy as np

import sklearn
true_news = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")

fake_news = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
true_news.head()

true_news.shape
true_news["True OR Fake "] = 1
fake_news['True OR Fake'] = 0
fake_news

fake_news.shape
frames = [true_news,fake_news]
true_news.columns = fake_news.columns
all_news_data = pd.concat([true_news, fake_news], ignore_index=True)
all_news_data.head()
X = all_news_data['text']

y = all_news_data['True OR Fake']
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(),MultinomialNB())
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(),LogisticRegression())
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))