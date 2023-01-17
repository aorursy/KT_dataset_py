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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report



df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding="latin-1")

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

X = df['v2']

y = df['label']

cv = CountVectorizer()

X = cv.fit_transform(X) # Fit the Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier

clf = MultinomialNB()

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
from sklearn.externals import joblib

joblib.dump(clf, 'NB_spam_model.pkl')

NB_spam_model = open('NB_spam_model.pkl','rb')

clf = joblib.load(NB_spam_model)