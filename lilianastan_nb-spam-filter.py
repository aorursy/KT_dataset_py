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
filename = '/kaggle/input/spam-ham-sms-dataset/sms_spam.csv'

df = pd.read_csv(filename, delimiter=',')

labels = df['type'].values

messages = df['text'].values

# print(messages[0])

spam_messages = 1 - np.sum(labels == 'ham') / labels.shape[0]

print('Percentage of spam messages: %.2f'%spam_messages, ' out of a total of ', labels.shape[0], 'messages')
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelBinarizer

shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)



for train_index, test_index in shuffle_stratified.split(messages, labels):

    msg_train, msg_test = messages[train_index], messages[test_index]

    labels_train, labels_test = labels[train_index], labels[test_index]



label_binarizer = LabelBinarizer()

label_binarizer.fit(labels_train)



y_train = label_binarizer.transform(labels_train)

y_test = label_binarizer.transform(labels_test)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB



count_vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english')



count_vectorizer.fit(msg_train)

X_train = count_vectorizer.transform(msg_train)

X_test = count_vectorizer.transform(msg_test)



model = MultinomialNB(alpha=0.01)

model.fit(X_train, y_train.ravel())



predictions = model.predict(X_test)
import random

from sklearn.metrics import classification_report

# print(classification_report(y_test, predictions))

select_index = random.randint(0, X_test.shape[0])

print('sms predicted class: ', model.predict(X_test[select_index]))

print('sms probability estimates: ', model.predict_proba(X_test[select_index]))



print('Actual message: \n', msg_test[select_index],

     '\nActual label: ', labels_test[select_index])

print('Reverse vectorizer: \n', count_vectorizer.inverse_transform(X_test[select_index]))