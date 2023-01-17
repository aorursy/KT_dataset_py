# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelEncoder

import string

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#1. Load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
# 2. Remove punctiation & convert to lowercase:

train['text_processed'] = train.text.str.replace('[{}]'.format(string.punctuation), '').str.lower()

test['text_processed'] = test.text.str.replace('[{}]'.format(string.punctuation), '').str.lower()
# 3. Words count vector:

counter = CountVectorizer(stop_words='english')

counter.fit(train.text_processed)

train_counts = counter.transform(train.text_processed)

test_counts = counter.transform(test.text_processed)
# 3. Change authors names' to numerical labels

le = LabelEncoder()

train_labels = le.fit_transform(train.author)

list(le.classes_)
# 4. Classify!

classifier = MultinomialNB()

classifier.fit(train_counts, train_labels)

predictions = classifier.predict_proba(test_counts)
submission = pd.DataFrame(np.hstack((test.id.values.reshape(-1, 1), predictions)), columns=['id', 'EAP', 'HPL', 'MWS'])

submission.to_csv('submission.csv',index=False)

submission.head()