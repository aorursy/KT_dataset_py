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
df = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv', encoding ='latin1')
df.head()
del df['Unnamed: 0']

del df['type']

del df['file']
df.head()
df['label'].value_counts()
df = df[df.label != 'unsup']
df['label'].value_counts()
df.isnull().sum()
blanks = []

for i, rv, lb in df.itertuples():

    if type(rv)== str:

        if rv.isspace():

            blanks.append(i)

print(len(blanks))
df.review.iloc[0]
from sklearn.model_selection import train_test_split
X = df['review']

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])

text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf_nb.fit(X_train, y_train)
prediction_nb = text_clf_nb.predict(X_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, prediction_nb))
print(metrics.classification_report(y_test, prediction_nb))
print(metrics.accuracy_score(y_test, prediction_nb))
text_clf_lsvc.fit(X_train, y_train)
prediction_lsvc = text_clf_lsvc.predict(X_test)
print(metrics.confusion_matrix(y_test, prediction_lsvc))
print(metrics.classification_report(y_test, prediction_lsvc))
print(metrics.accuracy_score(y_test, prediction_lsvc))