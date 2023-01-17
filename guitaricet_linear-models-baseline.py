import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn



import scipy

import sklearn.metrics as skm

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer



seaborn.set()

%matplotlib inline
!ls ..
# load data

train = pd.read_csv('../input/train.csv')

valid = pd.read_csv('../input/valid.csv')

test = pd.read_csv('../input/test.csv')



# remove NA

train.fillna('', inplace=True)

valid.fillna('', inplace=True)

test.fillna('', inplace=True)



# change label type to pd.category

train['label'] = train.label.astype('category')

valid['label'] = valid.label.astype('category')

test['label'] = valid.label.astype('category')
train.head()
train.label.value_counts()
tfidf_title = TfidfVectorizer(stop_words='english')

tfidf_title.fit(train.title)
model = LinearSVC()

model.fit(tfidf_title.transform(train.title), train.label.cat.codes)
predictions = model.predict(tfidf_title.transform(valid.title))

skm.f1_score(valid.label.cat.codes, predictions, average='macro')
predictions = model.predict(tfidf_title.transform(test.title))
mapping = test.label.cat.categories

predictions_str = [mapping[p] for p in predictions]

print(predictions[:5])

print(predictions_str[:5])
submission = pd.DataFrame({'id': test.index, 'label': predictions_str})

submission.to_csv('submission.csv', index=False)

submission.head()