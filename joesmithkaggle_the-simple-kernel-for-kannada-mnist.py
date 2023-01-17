import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import csv

import os

import warnings

warnings.filterwarnings('ignore')

import gc

gc.enable()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

tr_data = np.array(data).astype('int')

X_tr = tr_data[0:60000,1:]

y_tr = tr_data[0:60000,0]

del data, tr_data

gc.collect()
data = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

tr_tst_data = np.array(data).astype('int')

X_tst = tr_tst_data[0:10240,1:]

y_tst = tr_tst_data[0:10240,0]

del data, tr_tst_data

gc.collect()
data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

fl_tst_data = np.array(data).astype('int')

X = fl_tst_data[0:5000,1:]

del data, fl_tst_data

gc.collect()

print("Data is ready !!")
clf = SVC(kernel='rbf')
clf.fit(X_tr, y_tr)
predictions = clf.predict(X_tst)
print('The accuracy: ', accuracy_score(y_tst, predictions))

print(confusion_matrix(y_tst,predictions))

print(classification_report(y_tst,predictions))

del predictions

gc.collect()
predictions = clf.predict(X)

sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

sub['label'] = predictions

sub.to_csv('submission.csv', index=False)