# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import fetch_20newsgroups

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer



emails = fetch_20newsgroups()
print(emails.target_names)
emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])

emails.data[0]
emails.target_names
emails.target[0]

#This is a baseball email.
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], \

                                  subset = 'train', shuffle = True, \

                                  random_state = 108)

test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], \

                                  subset = 'test', shuffle = True, \

                                  random_state = 108)
counter = CountVectorizer()

counter.fit(train_emails.data + test_emails.data)

train_counts = counter.transform(train_emails.data)

test_counts = counter.transform(test_emails.data)

print(train_counts.shape)

print(test_counts.shape)

print(train_counts[0,:])
classifier = MultinomialNB()

classifier.fit(train_counts, train_emails.target)

print(classifier.score(test_counts, test_emails.target))



train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)

test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)



counter = CountVectorizer()

counter.fit(test_emails.data + train_emails.data)

train_counts = counter.transform(train_emails.data)

test_counts = counter.transform(test_emails.data)



classifier = MultinomialNB()

classifier.fit(train_counts, train_emails.target)



print(classifier.score(test_counts, test_emails.target))