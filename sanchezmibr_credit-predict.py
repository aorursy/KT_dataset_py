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
NOMES = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

dados = pd.read_csv("/kaggle/input/credit-screening.data",na_values = '?', names=nomes, sep=",")

dados.head()
dados.describe()
dados.isnull().sum()
dados.fillna({'A1':dados['A1'].mode(0),'A4':dados['A4'].mode(0), 'A5':dados['A5'].mode(0), 'A6':dados['A6'].mode(0), 'A7':data['A7'].mode(0)}, inplace= True)

dados
dados.fillna({'A2':dados['A2'].mean()})

dados.fillna({'A14':dados['A14'].mean()})

dados[{'A1', 'A4', 'A5', 'A6','A7','A9','A10','A12','A13'}].astype('category')
dados['A16'].astype('category').cat.codes
dados = pd.get_dummies(data, columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])

data.head()
dic = dt.load_digits()

dic.keys()
X = dic.data

y = dic.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)

y_train.shape

X_train.shape

y_test.shape

y_train.shape
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
