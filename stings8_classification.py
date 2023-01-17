import pandas as pd

import warnings

from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression

from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/data.csv')
df.head()
df.info()
df['diagnosis'].unique().tolist()
dg = {'M': 0, 'B' : 1}

df['diagnosis']= df['diagnosis'].map(dg)
corr = df.corr()

corr = corr.iloc[:,1]

corr = pd.DataFrame(corr, df.columns)

corr.sort_values(by=['diagnosis'], ascending=False)
df.hist(figsize=(15,15), bins=50);
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
X = df.drop(columns='diagnosis')

y = df.iloc[:,0]

X_test, X_Train, y_test, y_train = train_test_split(X, y, random_state=42, stratify=y)
std = StandardScaler()

std.fit(X_Train)

X_Train = std.transform(X_Train)

X_test = std.transform(X_test)
perc = Perceptron(max_iter = 1000 , tol = 1e-3)

perc.fit(X_Train, y_train)

pred_perc = perc.predict(X_test)

accuracy_score(y_test, pred_perc)
sgd = SGDClassifier(max_iter = 1000 , tol = 1e-3)

sgd.fit(X_Train, y_train)

pred_sgd = sgd.predict(X_test)

accuracy_score(y_test, pred_sgd)
logreg = SGDClassifier(max_iter = 1000)

logreg.fit(X_Train, y_train)

pred_log = logreg.predict(X_test)

accuracy_score(y_test, pred_log)