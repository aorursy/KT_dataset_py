import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', index_col = 'id')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', index_col = 'id')

submit = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv', index_col = 'id')

train.head()
train.isna().sum()/len(train)
test.isna().sum()/len(test)
y = train.target

train.drop(['target', 'location', 'keyword'], axis = 1, inplace = True)

test.drop(['location', 'keyword'] , axis = 1, inplace= True)
train.head()
import string

from string import digits

replace = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def remove_punct(words):

    remove_digits = str.maketrans('', '', digits) 

    words = words.translate(remove_digits)

    words = words.split()

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in words]

    return " ".join(stripped)



train.text = train.text.apply(remove_punct )

test.text = test.text.apply(remove_punct)
train.head()
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 1)
X_train.head()
cnt = CountVectorizer(ngram_range= (1,1))

X_train = cnt.fit_transform(X_train.text)

X_test = cnt.transform(X_test.text)

test = cnt.transform(test.text)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', C=0.3)

scores = cross_val_score(clf, X_train, y_train, cv=3)

scores
clf.fit(X_train , y_train)

y_pre = clf.predict(X_test)

sum(y_test == y_pre)/ len(y_test)
y_submit = clf.predict(test)
print(y_submit)
submit.head()
submit.target = y_submit
submit.target
submit.to_csv('submit.csv')