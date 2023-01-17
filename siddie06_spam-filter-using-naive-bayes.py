# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/emails.csv')
data.head()
data.info()
X = data['text']

y = data['spam']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state = 324)
vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(X_train.values)

classifier = MultinomialNB()

targets = y_train.values

classifier.fit(counts,targets)
pred_counts = vectorizer.transform(X_test.values)
predictions = classifier.predict(pred_counts)
acc = accuracy_score(y_test,predictions)
acc