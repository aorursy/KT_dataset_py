import numpy as np 

import pandas as pd

import os

import sklearn.datasets as dt

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

data = pd.read_csv("/kaggle/input/credit-screening.data",na_values = '?', names=nomes, sep=",")

data.head()
data.describe()
data.isnull().sum()
data.fillna({'A1':data['A1'].mode(0),'A4':data['A4'].mode(0), 'A5':data['A5'].mode(0), 'A6':data['A6'].mode(0), 'A7':data['A7'].mode(0)}, inplace= True)

data
data.fillna({'A2':data['A2'].mean()})

data.fillna({'A14':data['A14'].mean()})

data[{'A1', 'A4', 'A5', 'A6','A7','A9','A10','A12','A13'}].astype('category')

data['A16'].astype('category').cat.codes

data = pd.get_dummies(data, columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])

data.head()
dic = dt.load_digits()

dic.keys()
X = dic.data

y = dic.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)
y_train.shape

X_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn

model = knn.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_score = model.score(X_test, y_test)

y_pred

y_score
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))