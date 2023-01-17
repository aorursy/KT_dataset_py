import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv('/kaggle/input/news.csv')

df.head()
df.shape
df.tail()
df.isnull().sum()
X = df['text']

y = df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=1)
vec= TfidfVectorizer(stop_words='english', max_df=0.7)

vec_train = vec.fit_transform(X_train)

vec_test  = vec.transform(X_test)

agg = PassiveAggressiveClassifier(max_iter=50)

agg.fit(vec_train,y_train)
y_pred = agg.predict(vec_test)
accuracy_score(y_test,y_pred)*100
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])