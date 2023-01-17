import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt
costarican = pd.read_csv(r"../input/costa-rican-household-poverty-prediction/train.csv")
costaricanTest = pd.read_csv(r"../input/costa-rican-household-poverty-prediction/test.csv")
costarican.shape
costarican.head()
costaricanTest.head()
print(costarican.isnull().sum())
XCR = costarican.drop(columns=['Id','v2a1','v18q1','rez_esc','idhogar','Target'])

XCR.head()
XCRtest = costaricanTest.drop(columns=['Id','v2a1','v18q1','rez_esc','idhogar'])

XCRtest.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
XCR.shape
YCR = costarican.Target
XCR.edjefe = XCR.edjefe.eq('yes').mul(1)

XCR.dependency = XCR.dependency.eq('yes').mul(1)

XCR.edjefa = XCR.edjefa.eq('yes').mul(1)
XCRtest.edjefe = XCRtest.edjefe.eq('yes').mul(1)

XCRtest.dependency = XCRtest.dependency.eq('yes').mul(1)

XCRtest.edjefa = XCRtest.edjefa.eq('yes').mul(1)
XCR.edjefe = XCR.edjefe.eq('no').mul(0)

XCR.dependency = XCR.dependency.eq('no').mul(0)

XCR.edjefa = XCR.edjefa.eq('no').mul(0)
XCRtest.edjefe = XCRtest.edjefe.eq('no').mul(0)

XCRtest.dependency = XCRtest.dependency.eq('no').mul(0)

XCRtest.edjefa = XCRtest.edjefa.eq('no').mul(0)
for a in XCR.columns:

    XCR[a].fillna(XCR[a].mode()[0],inplace=True)
for b in XCRtest.columns:

    XCRtest[b].fillna(XCRtest[b].mode()[0],inplace=True)
best = 0

media_anterior = 0

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors = i)

    scores = cross_val_score(knn, XCR, YCR, cv=10)

    media = sum(scores)/len(scores)

    if media > media_anterior:

        media_anterior = media

        best = i

print(media_anterior)

print(best)
knn = KNeighborsClassifier(n_neighbors = best)
knn.fit(XCR,YCR)
YtestPred = knn.predict(XCRtest)
Pred = pd.DataFrame(YtestPred)

Pred.columns = ['Target']

Pred.insert(0, 'Id', costaricanTest.Id , True)

Pred.head()
Pred.to_csv('CRPrediction.csv', header=True, index=False)