import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt
fashion_train = pd.read_csv('../input/fashion-mnist_train.csv')

fashion_test = pd.read_csv('../input/fashion-mnist_test.csv')
!pip install idx2numpy

import idx2numpy
file = '../input/t10k-images-idx3-ubyte'
array = idx2numpy.convert_from_file(file)
print(array[2])
plt.imshow(array[2], cmap = 'gray')

plt.show()
from sklearn.model_selection import train_test_split

df = fashion_train

x = df.drop(['label'], axis = 1)

y = df.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 25, max_depth = 12, random_state = 2)

model.fit(x_train, y_train)
pred = model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))