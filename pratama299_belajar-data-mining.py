# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from seaborn import pairplot

data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
pairplot(data,hue="species")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop("species",axis=1),data["species"],test_size=0.3)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(x_train, y_train) #si mesin learningnya disini

clf
hasil_prediksi = clf.predict(x_test) #untuk Prediksi

hasil_prediksi
from sklearn.metrics import accuracy_score
accuracy_score(y_test,hasil_prediksi)
data.head()