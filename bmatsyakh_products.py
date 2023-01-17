# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_products = pd.read_csv("../input/train.csv")
df_products.head()
df_y = df_products["target"]
df_y.head()
df_X = df_products.drop('id', axis=1).drop('target', axis=1)
df_X.head()
df_X.to_csv("train_X.csv", header=True, index=False)

df_y.to_csv("train_y.csv", header=True, index=False)
pd.read_csv("train_X.csv")
X = df_X.values
X
y = df_y.values
y = y.reshape(-1)
print("y shape:", y.shape)
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X train:", X_train.shape)

print("y train:", y_train.shape)

print()

print("X test: ", X_test.shape)

print("y test: ", y_test.shape)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=45, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
accuracy_train = accuracy_score(y_train, y_train_pred)

accuracy_test = accuracy_score(y_test, y_test_pred)
print("Training accuracy: {0:.3f}%".format(accuracy_train * 100))

print("Test accuracy: {0:.3f}%".format(accuracy_test * 100))