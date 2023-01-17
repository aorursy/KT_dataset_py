import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X = train.drop('label', 1)

Y = train['label']
print(X.isnull().count())

print(Y.isnull().count())
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, shuffle=False)
clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(train_X, train_Y)
accuracy = clf.score(test_X, test_Y)
target = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

predict = clf.predict(target)
sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sub['Label'] = predict

sub.to_csv('my_submission.csv', index=False)