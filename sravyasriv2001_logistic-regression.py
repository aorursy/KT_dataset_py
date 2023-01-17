import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv('../input/framingham.csv')

dataset.head(100)
dataset=dataset.dropna()

dataset.info
X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score as score

print('Accuracy')

print(score(y_test,y_pred)*100)