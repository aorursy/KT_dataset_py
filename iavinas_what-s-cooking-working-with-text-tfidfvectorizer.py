import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

%matplotlib inline

import os

print(os.listdir("../input"))
train = pd.read_json('../input/train.json')

train.set_index('id' , inplace= True)

label = train['cuisine']

train.drop('cuisine' , axis = 1 , inplace= True)

test = pd.read_json('../input/test.json')
train.head()
test.head()
print('Number of train data ' , len(train))

print('Number of test data ' , len(test))
len(label.unique())
plt.figure(figsize=(16, 6))

sns.countplot(y = label , order = label.value_counts().index)
type(train.ingredients[0])
print('Maximum ingredients used in a single cuisine' , train.ingredients.apply(len).max())

print('Minimum ingredients used in a single cuisine' , train.ingredients.apply(len).min())
def list_to_text(data):

    return (" ".join(data)).lower()
list_to_text(['a' , 'b'])
train.ingredients = train.ingredients.apply(list_to_text )

test.ingredients = test.ingredients.apply(list_to_text)
train.head()
test.head()


tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train.ingredients)

X_test = tfidf.transform(test.ingredients)
l = LabelEncoder()

label = l.fit_transform(label)
label
clf = XGBClassifier()

scores = cross_val_score(clf, X_train, label, cv=3).mean()

scores
clf.fit(X_train , label)

pre = clf.predict(X_test)
pre
pre = l.inverse_transform(pre)

pre
submit = pd.read_csv('../input/sample_submission.csv')

submit.head()
submit.cuisine = pre

submit.id = test.id
submit.to_csv('submit.csv' , index= False)
!ls