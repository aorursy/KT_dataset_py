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
from sklearn.datasets import make_circles

import matplotlib.pyplot as plt

from numpy import where
X, y = make_circles(n_samples = 1000, noise =0.1, random_state =1)
for i in range(2):

    samples_ix = where(y==i)

    plt.scatter(X[samples_ix, 0], X[samples_ix, 1])

plt.show()
# split into train and test

n_test = 500

trainX, testX = X[:n_test, :], X[n_test:, :]

trainy, testy = y[:n_test], y[n_test:]
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(100, input_dim=2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model

history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)

# evaluate the model

_, train_acc = model.evaluate(trainX, trainy, verbose=0)

_, test_acc = model.evaluate(testX, testy, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# predict probabilities for test set

yhat_probs = model.predict(testX, verbose=0)

# predict crisp classes for test set

yhat_classes = model.predict_classes(testX, verbose=0)
# reduce to 1d array

yhat_probs = yhat_probs[:, 0]

yhat_classes = yhat_classes[:, 0]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(testy, yhat_classes)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(testy, yhat_classes)

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(testy, yhat_classes)

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(testy, yhat_classes)

print('F1 score: %f' % f1)
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(testy, yhat_classes)

print(matrix)
from sklearn.metrics import roc_auc_score

# ROC AUC

auc = roc_auc_score(testy, yhat_probs)

print('ROC AUC: %f' % auc)