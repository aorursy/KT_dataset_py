# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron # TODO try MLP multilayer perceptron

clf = Perceptron(tol=1e-3, random_state=0)

X, y = load_digits(return_X_y=True)
print(len(X))
print(len(X[99]))
print(y[99])
# 28x28
# 8x8
for row in range(0,8):
    for column in range(0,8):
        value = X[99][(8 * row) + column]
        letter = "." if value == 0 else "X" #TERNARY
        print(letter, end=" ")
    print("")

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)

#TRAIN
clf.fit(X_train, y_train)

# TODO split into test imported or slice :  100: :-100
#TRAIN
# clf.fit(X, y)

#TEST
# print(clf.__dict__)
clf.score(X_test, y_test)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

clf = MLPClassifier(random_state=1, hidden_layer_sizes=800, max_iter=100)

# X, y = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)

#TRAIN
clf.fit(X_train, y_train)

#TEST
clf.score(X, y)
# print(clf.__dict__)
