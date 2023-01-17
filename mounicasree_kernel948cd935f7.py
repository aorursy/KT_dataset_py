# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
spam = pd.read_csv('//kaggle//input//sms-spam-collection-dataset//spam.csv',encoding = "latin-1",usecols=['v1','v2'])

spam.head()
spam_text = spam.v2

spam_label = spam.v1
spam_label[spam_label.isnull()]


spam_text_train,spam_text_test,spam_label_train,spam_label_test = train_test_split(spam_text,spam_label,test_size=0.3,random_state=42)

spam_text_train.shape,spam_text_test.shape,spam_label_train.shape,spam_label_test.shape


count_vec = CountVectorizer(stop_words="english")
count_train = count_vec.fit_transform(spam_text_train)

count_train.shape
count_test = count_vec.transform(spam_text_test)

count_test.shape


nb = MultinomialNB()
nb.fit(count_train,spam_label_train)

spam_pred=nb.predict(count_test)


accuracy_score(spam_label_test,spam_pred)
classification_report(spam_label_test,spam_pred)
spam_label_test
spam_pred