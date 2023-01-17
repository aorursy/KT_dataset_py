import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score,r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.svm import SVR

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
trainX = pd.read_csv('/kaggle/input/mlcoursemm2020spring/trainX.csv')

trainY = pd.read_csv('/kaggle/input/mlcoursemm2020spring/trainY.csv')

testX = pd.read_csv('/kaggle/input/mlcoursemm2020spring/testX.csv')

del trainX['Id']

del testX['Id']
trainX = (trainX - trainX.mean())/trainX.std()

testX = (testX - testX.mean())/testX.std()
X = trainX.copy()

y = trainY['Value'].copy()

for i in range(1,10):

    reg = SVR().fit(X,y)

    pred = reg.predict(X)

    bad2 = set()

    bad2 |= set(X[abs(pred-y) > 100/i].index)

    X = X.drop(bad2)

    y = y.drop(bad2)

    if X.shape[0]<=500:

        break

    print(X.shape[0])

for i in range(1,500):

    reg = Ridge().fit(X,y)

    pred = reg.predict(X)

    bad2 = set()

    bad2 |= set(X[abs(pred-y) > 100/i].index)

    X = X.drop(bad2)

    y = y.drop(bad2)

print(X.shape[0])
reg = Ridge().fit(X,y)

pred = reg.predict(testX)

#pred += 32 - pred.mean()

ans = pd.DataFrame({'Value' : pred})

ans['Id'] = range(len(ans))

ans.mean()
ans.to_csv('solution16_05_2020_8_48.csv',index=False)