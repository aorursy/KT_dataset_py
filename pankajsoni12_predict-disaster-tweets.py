# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
print(train.head(2))

print(test.head(2))

print(train.shape)

print(test.shape)

print(train.info())

print(test.info())
plt.figure(figsize=(25,4))

sns.countplot(train.keyword)

plt.xticks(rotation=90)

plt.show()
train.keyword.unique().size
train.keyword.mode()
train.keyword.value_counts()
train.location.unique().size
train.isnull().any()
train.isnull().sum()
train.text.head(5)
train.text[90]
import re
def textCleaning(text):

    text = re.sub("http://.*","",text)

    text = re.sub("[^\w\s]+","",text)

    text = re.sub("[\d]+","",text)

    text = re.sub("\s+"," ",text)

    text = re.sub("^\s","",text)

    text = re.sub("\s$","",text)

    #print(text)

    return(text)

# textCleaning(train.text[90])
train.text = train.text.apply(textCleaning)
test.text = test.text.apply(textCleaning)
train.text[90]
n_gram = (1,4)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split
# cv = CountVectorizer(stop_words="english",ngram_range=n_gram)

cv = CountVectorizer(ngram_range=n_gram)
from sklearn.naive_bayes import GaussianNB,MultinomialNB
X_train,X_test,y_train,y_test = train_test_split(train.text,train.target,test_size=.2,random_state=5)
X_train_dtm = cv.fit_transform(X_train).toarray()

X_test_dtm = cv.transform(X_test).toarray()
nb = GaussianNB()

nb.fit(X_train_dtm,y_train)

y_pred = nb.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
mnb = MultinomialNB()

mnb.fit(X_train_dtm,y_train)

y_pred = mnb.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train_dtm,y_train)

y_pred = lr.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()

lgbm.fit(X_train_dtm,y_train)

y_pred = lgbm.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
from catboost import CatBoostClassifier
cb = CatBoostClassifier()

cb.fit(X_train_dtm,y_train)

y_pred = cb.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
# tfidf = TfidfVectorizer(stop_words="english",ngram_range=n_gram)

tfidf = TfidfVectorizer(ngram_range=n_gram)

X_train_dtm = tfidf.fit_transform(X_train).toarray()

X_test_dtm = tfidf.transform(X_test).toarray()
nb = GaussianNB()

nb.fit(X_train_dtm,y_train)

y_pred = nb.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
lr = LogisticRegression()

lr.fit(X_train_dtm,y_train)

y_pred = lr.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()

lgbm.fit(X_train_dtm,y_train)

y_pred = lgbm.predict(X_test_dtm)

accuracy_score(y_test,y_pred)
from catboost import CatBoostClassifier
cb = CatBoostClassifier()

cb.fit(X_train_dtm,y_train)

y_pred = cb.predict(X_test_dtm)

accuracy_score(y_test,y_pred)