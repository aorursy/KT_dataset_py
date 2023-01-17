import numpy as np

import pandas as pd
train = pd.read_csv('salary-train.csv', header=0)

train.head()
train['FullDescription'] = train['FullDescription'].apply(str.lower)
train.head()
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=5)

fd_train = vectorizer.fit_transform(train['FullDescription'])
train['LocationNormalized'].fillna('nan', inplace=True)

train['ContractTime'].fillna('nan', inplace=True)
test = pd.read_csv('salary-test-mini.csv', header=0)

test.head()
test.drop('SalaryNormalized', axis=1, inplace=True)

test.head()
test['FullDescription'] = test['FullDescription'].apply(str.lower)

test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

fd_test = vectorizer.transform(test['FullDescription'])

test['LocationNormalized'].fillna('nan', inplace=True)

test['ContractTime'].fillna('nan', inplace=True)
from sklearn.feature_extraction import DictVectorizer

enc = DictVectorizer()

X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
type(X_train_categ)
X_train_categ.shape
type(fd_train)
fd_train.shape
fd_train.ndim
X_train_categ.ndim
from scipy.sparse import hstack

X_train = hstack([fd_train, X_train_categ])

X_test = hstack([fd_test, X_test_categ])
X_test
X_train
y_train = train['SalaryNormalized']
from sklearn.linear_model import Ridge
linreg = Ridge(alpha=1, random_state=241)

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
y_pred
[round(x, 2) for x in y_pred]