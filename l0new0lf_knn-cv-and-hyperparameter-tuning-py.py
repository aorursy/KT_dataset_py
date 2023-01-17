import numpy as np

np.random.seed(123)

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



import warnings

warnings.filterwarnings('ignore')
pixels = pd.read_csv('../input/toysets/6.overlap.csv',  names=['x1', 'x2', 'y'])

ys = pixels[['y']]

Xs = pixels.drop(['y'], axis=1)



print(Xs.shape)

print(ys.shape)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
# train, cv, test

Xs, X_test, ys, y_test = train_test_split(Xs, ys, test_size=0.3)

X_tr, X_cv, y_tr, y_cv = train_test_split(Xs, ys, test_size=0.3)
hist = {

    'ks': [],

    'acc_cv': [],

    'acc_tr': []

}



for k in range(1,50,2):

    knn = KNeighborsClassifier(n_neighbors=k)

    # fitting cv train

    knn.fit(X_tr, y_tr)

    # predict and eval  cv train

    pred_cv = knn.predict(X_cv)

    pred_tr = knn.predict(X_tr)

    acc_cv = accuracy_score(y_cv, pred_cv, normalize=True) * float(100)

    acc_tr  = accuracy_score(y_tr, pred_tr, normalize=True) * float(100)

    print(f"k:{k}\t val-acc: {acc_cv} \ttrain-acc: {acc_tr}")

    

    # log

    hist['ks'].append(k)    

    hist['acc_cv'].append(acc_cv)    

    hist['acc_tr'].append(acc_tr)
plt.figure(figsize=(20, 7))



plt.plot(hist['ks'], hist['acc_cv'], label='cv')

plt.plot(hist['ks'], hist['acc_tr'], label='train')



for k, acc_cv in zip(hist['ks'], hist['acc_cv']):

    plt.text(k, acc_cv, f'k={k}')



plt.legend()

plt.show()
k = 31



knn = KNeighborsClassifier(n_neighbors=k)

# fitting cv train

knn.fit(X_tr, y_tr)

# predict and eval  cv train

pred_cv = knn.predict(X_cv)

acc_cv = accuracy_score(y_cv, pred_cv, normalize=True) * float(100)

print(f"k:{k}\t val-acc: {acc_cv}")
from sklearn.model_selection import cross_val_score
hist = {

    'ks': [],

    'acc_cv': []

}



for k in range(0, 60):

    knn = KNeighborsClassifier(n_neighbors=k) 

    scores = cross_val_score(knn, Xs, ys, cv=10) # 10-fold

    # 95% conf-interval: scores.mean() (+/- 2*scores.std())

    

    # log

    hist['ks'].append(k)

    hist['acc_cv'].append(scores.mean())
plt.figure(figsize=(20, 7))



plt.plot(hist['ks'], hist['acc_cv'], label='cv')



for k, acc_cv in zip(hist['ks'], hist['acc_cv']):

    plt.text(k, acc_cv, f'k={k}')



plt.legend()

plt.show()