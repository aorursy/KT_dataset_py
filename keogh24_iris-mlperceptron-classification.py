# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('../input/Iris.csv')
df.head()
df.describe().transpose()
df2 = df.drop(['Id'], axis=1)
df2['Species'].replace(to_replace=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], value=[1., 2., 3.], inplace=True)
df2.head()
y = df2[['Species']]
y.head()
X = df2[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
nn = MLPClassifier(activation='relu', solver='sgd', hidden_layer_sizes=(10, 10, 10), max_iter=1000)
nn.fit(X_train, y_train.values.ravel())
predictions = nn.predict(X_test)
predictions
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
