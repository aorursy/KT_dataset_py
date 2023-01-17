import pandas as pd

import numpy as np

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

data = data.drop(columns={'Unnamed: 2','Unnamed: 4','Unnamed: 3'}, axis=1)
data['v1'] = data['v1'].apply(lambda x: 1 if x=='spam' else 0)
data
X_train,X_test,y_train,y_test = train_test_split(data['v2'],data['v1'].values)
tfid = TfidfVectorizer(decode_error='ignore')

_xtrain = tfid.fit_transform(X_train)

_xtest = tfid.transform(X_test)
model = MultinomialNB()

model.fit(_xtrain,y_train)
model = AdaBoostClassifier()

model.fit(_xtrain,y_train)
model.score(_xtest,y_test)