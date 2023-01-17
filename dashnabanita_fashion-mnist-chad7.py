# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
fashion_train = pd.read_csv('../input/fashion-mnist_train.csv')

fashion_test = pd.read_csv('../input/fashion-mnist_test.csv')

print(fashion_train.shape)

print(fashion_test.shape)
print(fashion_train.head())

print(fashion_test.head())

import matplotlib.pyplot as plt

import matplotlib as mpl
fashion_train.keys()

fashion_test.keys()
fashion_train.info()
fashion_train.describe()
X_train = fashion_train.iloc[:,1:]

X_test = fashion_test.iloc[:,1:]

y_train = fashion_train['label']

y_test = fashion_test['label']
print(y_test)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
image = X_train[20345]

image = image.reshape(28,28)

plt.imshow(image, cmap = mpl.cm.binary, interpolation= "nearest")

plt.axis("off")

plt.show()
y_train[20345]
fashion_train.iloc[:,0].value_counts()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")
y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)

imputer.fit(X_train)

imputer.fit(y_train)
for i in range(0,784):

    imputer.transform(X_train[:,i].reshape(-1, 1))

    imputer.transform(X_test[:,i].reshape(-1, 1))

imputer.transform(y_train)

imputer.transform(y_test)
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
std_scaler.fit_transform(X_train)

std_scaler.fit_transform(X_test)

std_scaler.fit_transform(y_train)

std_scaler.fit_transform(y_test)
X_train = pd.DataFrame(X_train)

X_test = pd.DataFrame(X_test)

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)

correlations = fashion_train.corr()['label'].sort_values()
correlations.tail(10)

correlations.head(10)
y_train_s = (y_train == 2)

y_test_s = (y_test == 2)

y_train_s = y_train_s.values.reshape(-1,1)

y_test_s = y_test_s.values.reshape(-1, 1)
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state = 42)

sgd_clf.fit(X_train,y_train_s)
image1 = X_train.values[20345]

sgd_clf.predict([image1])
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train,y_train_s, cv = 10, scoring="accuracy")
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_s,cv=10)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_s, y_train_pred)
y_scores = sgd_clf.decision_function([image1])

print(y_scores)

threshold = 0

y_image1_pred = (y_scores > threshold)

print(y_image1_pred)