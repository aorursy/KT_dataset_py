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
train_df = pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv')

test_df = pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv')



train_df.head()
import matplotlib.pyplot as plt



plt.hist(train_df['rating']);
# extract only review as feature, rating as target

X_train = train_df['review']

y_train = train_df['rating']

X_test = test_df['review']

y_test = test_df['rating']
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



# ngram_range=(1,2) so that the model can deal with text such as "not good" as one word

vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))

X_train = vect.fit_transform(X_train.tolist())

X_test = vect.transform(X_test.tolist())



nb = MultinomialNB().fit(X_train, y_train)



train_pred = nb.predict(X_train)

test_pred = nb.predict(X_test)



print('Training Accuracy:', accuracy_score(y_train, train_pred))

print('Testing Accuracy:', accuracy_score(y_test, test_pred))
# only use rather extreme rating as predicting standard

train_df['new_rating'] = train_df[(train_df['rating'] > 7) | (train_df['rating'] < 4)]['rating']

test_df['new_rating'] = test_df[(test_df['rating'] > 7) | (test_df['rating'] < 4)]['rating']



# 1 is good rating, 0 is bad rating

train_df['new_rating'] = train_df['new_rating'].apply(lambda x: 1 if x > 7 else 0)

test_df['new_rating'] = test_df['new_rating'].apply(lambda x: 1 if x > 7 else 0)



train_df.head()
X_train = train_df['review']

X_test = test_df['review']

y_train = train_df['new_rating']

y_test = test_df['new_rating']

vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))

X_train = vect.fit_transform(X_train.tolist())

X_test = vect.transform(X_test.tolist())



nb = MultinomialNB().fit(X_train, y_train)

train_pred = nb.predict(X_train)

test_pred = nb.predict(X_test)



print('Training Accuracy:', accuracy_score(y_train, train_pred))

print('Testing Accuracy:', accuracy_score(y_test, test_pred))
alphas = np.array([0.001, 0.01, 0.1, 0, 1, 10, 100])



train_accu = []

test_accu = []



for alpha in alphas:

    nb = MultinomialNB(alpha=alpha).fit(X_train, y_train)

    train_pred = nb.predict(X_train)

    test_pred = nb.predict(X_test)

    

    train_accu.append(accuracy_score(y_train, train_pred))

    test_accu.append(accuracy_score(y_test, test_pred))



print('Training Accuracies')

print(train_accu)

print('Testing Accuracies')

print(test_accu)
plt.figure(figsize=(10, 6))

plt.plot(list(range(len(alphas))),train_accu, label='Training');

plt.plot(list(range(len(alphas))), test_accu, label='Testing');

plt.xticks(list(range(len(alphas))), alphas);

plt.xlabel('Alpha');

plt.ylabel('Accuracy')

plt.legend();