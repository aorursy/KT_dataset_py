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
import pandas as pd

ss = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
import matplotlib.pyplot as plt

import matplotlib as mlt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
train.head()
train[train["target"] == 1].head()
def null(df):

    null_percent = df.isnull().mean()*100

    null_val = df.isnull().sum()

    null_val_per = pd.concat([null_val, null_percent], axis=1, 

                             keys=["values", "percent"])

    print(null_val_per)
null(train)
null(test)
train[train["target"] == 1]["text"].values[1]
train[train["target"] == 0]["text"].head(1)
vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = vectorizer.fit_transform(train["text"])

test_vectors = vectorizer.fit_transform(test["text"])
train_vectors
test_vectors
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train["target"],

                                         cv=3, scoring="f1")



scores
clf.fit(train_vectors, train["target"])
X = train["text"]

y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
vect = CountVectorizer(stop_words='english')



X_train_cv = vect.fit_transform(X_train)

X_test_cv = vect.transform(X_test)
clf = MultinomialNB()

clf.fit(X_train_cv, y_train)
pred = clf.predict(X_test_cv)
confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
y_test = test["text"]

y_test_cv = vect.transform(y_test)

ss["target"] = clf.predict(y_test_cv)
ss.head()
from sklearn.linear_model import LogisticRegression
X = train["text"]

y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
tfv = TfidfVectorizer(analyzer="word", token_pattern=r'\w{1,}', stop_words="english")
tfv.fit((list(X_train)) + list(X_test))

xtrain_tfv = tfv.transform(X_train)

xtest_tfv = tfv.transform(X_test)
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV

from sklearn.svm import SVC
ss_1 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

X = train["text"]

y = train["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, 

                                                  random_state=42)
tfidf = TfidfVectorizer(stop_words="english")

X_train = tfidf.fit_transform(X_train)

X_val = tfidf.transform(X_val)
parameters = {

    "gamma":[0.7, 1, "auto", "scale"]

}



model = GridSearchCV(SVC(kernel="rbf"), parameters, cv=4, n_jobs=-1).fit(X_train, 

                                                                         y_train)
y_val_pred = model.predict(X_val)

accuracy_score(y_val, y_val_pred)
X_test = test["text"]

X_test = tfidf.transform(X_test)

y_test_pred = model.predict(X_test)
ss_1["target"] = y_test_pred

ss_1.to_csv("submission.csv", index=False)