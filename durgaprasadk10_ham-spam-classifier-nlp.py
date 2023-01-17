# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/spam-filter/emails.csv')
df.head()
df.spam.value_counts()
sns.countplot(x='spam',data=df)
df.info()
df.isna().sum()
df.isnull().sum()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.text,df.spam,test_size=0.2,random_state=42)
print(X_train.shape)

print(y_train.shape)
print(X_test.shape)

print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit_transform(X_train)

print(f"the total vocabulary size in the train data is {len(vectorizer.vocabulary_)}")
vectorizer.vocabulary_

#vocabulary:index
vectorizer.get_feature_names()
X_train_ct = vectorizer.transform(X_train)
X_train_ct.shape
X_train_ct[0]
print(X_train_ct[0])
X_train[0]
X_test_ct =  vectorizer.transform(X_test)
print(X_test_ct[0].shape)
print(X_test_ct[0])
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train_ct,y_train)
y_pred = knn_classifier.predict(X_test_ct)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train_ct,y_train)

y_pred = svc.predict(X_test_ct)

accuracy_score(y_test,y_pred)
from sklearn.naive_bayes import MultinomialNB

nb= MultinomialNB()

nb.fit(X_train_ct,y_train)

y_pred = nb.predict(X_test_ct)

accuracy_score(y_test,y_pred)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

vectorizer.fit(X_train)

X_train_ct  = vectorizer.transform(X_train)

X_test_ct = vectorizer.transform(X_test)
knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train_ct,y_train)

y_pred = knn_classifier.predict(X_test_ct)

accuracy_score(y_test,y_pred)
svc = SVC()

svc.fit(X_train_ct,y_train)

y_pred = svc.predict(X_test_ct)

accuracy_score(y_test,y_pred)
nb= MultinomialNB()

nb.fit(X_train_ct,y_train)

y_pred = nb.predict(X_test_ct)

accuracy_score(y_test,y_pred)
vectorizer = TfidfVectorizer(ngram_range=(1,2))

vectorizer.fit(X_train)

X_train_ct = vectorizer.transform(X_train)

X_test_ct = vectorizer.transform(X_test)
knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train_ct,y_train)

y_pred = knn_classifier.predict(X_test_ct)

accuracy_score(y_test,y_pred)
svc = SVC()

svc.fit(X_train_ct,y_train)

y_pred = svc.predict(X_test_ct)

accuracy_score(y_test,y_pred)
nb= MultinomialNB()

nb.fit(X_train_ct,y_train)

y_pred = nb.predict(X_test_ct)

accuracy_score(y_test,y_pred)