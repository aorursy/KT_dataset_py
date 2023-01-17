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
import tensorflow as tf
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
test_data
Y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X, Y)
predict = model.predict(X)
from sklearn.metrics import recall_score, precision_score, f1_score
recall_score(Y, predict)
precision_score(Y, predict)
f1_tree = f1_score(Y, predict)
f1_tree
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, Y)
predict = model.predict(X)
recall_score(Y, predict)
precision_score(Y, predict)
f1_gaussian = f1_score(Y, predict)
f1_gaussian
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(X, Y)
predict = model.predict(X)
recall_score(Y, predict)
precision_score(Y, predict)
f1_neural = f1_score(Y, predict)
f1_neural
average_F = (f1_tree + f1_gaussian + f1_neural)/3
average_F
