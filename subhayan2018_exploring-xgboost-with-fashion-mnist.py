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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import time

%matplotlib inline
fashion_mnist_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

fashion_mnist_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
x_train = np.array(fashion_mnist_train.iloc[:,1:])

y_train = np.array(fashion_mnist_train.iloc[:,0])

x_test = np.array(fashion_mnist_test.iloc[:,1:])

y_test = np.array(fashion_mnist_test.iloc[:,0])
print(x_train.shape, type(x_train))

print(y_train.shape, type(y_train))

print(x_test.shape, type(x_train))

print(y_test.shape, type(y_train))
plt.imshow(x_train[1].reshape((28,28)))

plt.show()
x_train = x_train/255.0

x_test = x_test/255.0
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(max_iter=2000)

logReg.fit(x_train,y_train)
prediction_test  = logReg.predict(x_test)

prediction_train = logReg.predict(x_train)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from sklearn.metrics import classification_report
print(accuracy_score(y_train, prediction_train))
print(accuracy_score(y_test, prediction_test))
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=300, n_jobs=-1, seed=0)

xgb_clf.fit(x_train,y_train)
y_pred = xgb_clf.predict(x_test)

print(accuracy_score(y_test, y_pred))
start = time.time()

xgb_clf = XGBClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.5, seed=0)

xgb_clf.fit(x_train,y_train)

stop = time.time()

print(xgb_clf.get_params)

print(f"Training time: {stop - start}s")
y_pred = xgb_clf.predict(x_test)

print(accuracy_score(y_test, y_pred))
start = time.time()

xgb_clf = XGBClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.5, gamma=2, seed=0)

xgb_clf.fit(x_train,y_train)

stop = time.time()

print(xgb_clf.get_params)

print(f"Training time: {stop - start}s")

y_pred = xgb_clf.predict(x_test)

print("test accuracy")

print(accuracy_score(y_test, y_pred))
start = time.time()

xgb_clf = XGBClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.5, max_deth= 5, min_child_weight= 1, seed=0)

xgb_clf.fit(x_train,y_train)

stop = time.time()

print(xgb_clf.get_params)

print(f"Training time: {stop - start}s")

y_pred = xgb_clf.predict(x_test)

print("test accuracy")

print(accuracy_score(y_test, y_pred))
start = time.time()

xgb_clf = XGBClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.5, max_deth= 5, min_child_weight= 1, seed=0)

xgb_clf.fit(x_train,y_train,eval_metric='mlogloss', early_stopping_rounds = 10, eval_set = [(x_test, y_test)])

stop = time.time()

print(xgb_clf.get_params)

print(f"Training time: {stop - start}s")

y_pred = xgb_clf.predict(x_test)

print("test accuracy")

print(accuracy_score(y_test, y_pred))
start = time.time()

xgb_clf = XGBClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.5, max_deth= 5, min_child_weight= 1, reg_lambda =20, seed=0)

xgb_clf.fit(x_train,y_train)

stop = time.time()

print(xgb_clf.get_params)

print(f"Training time: {stop - start}s")

y_pred = xgb_clf.predict(x_test)

print("test accuracy")

print(accuracy_score(y_test, y_pred))