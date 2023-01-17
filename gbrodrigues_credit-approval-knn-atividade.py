import os

for dirname, _, filenames in os.walk('/kaggle/input/credit/credit-screening.data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import numpy as np 

import pandas as pd

import os

import sklearn.datasets as dt

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()
import pandas as pd

nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

df = pd.read_csv("/kaggle/input/credit/credit-screening.data", sep = ",", names = nomes, na_values = '?')

df.head()
df.describe()
df.isnull().sum()
lista = ['A1','A4','A5','A6','A7']

for i in lista: 

    df[i]=df[i].fillna(value=df[i].mode)
lista = ['A2','A14']

for i in lista: 

    df[i]=df[i].fillna(value=df[i].mean)
lista = ['A1','A4','A5','A6','A7','A9','A10','A12','A13']

for i in lista:

    df[i] = df[i].astype(str)

    df[i] = df[i].astype('category')
df['A16'] = df['A16'].astype('category')

df['A16'].cat.codes
df = pd.get_dummies(df,columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
dic = dt.load_digits()

dic.keys()
X = dic.data

y = dic.target



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

modelo = knn.fit(X_train,y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test,y_test)

y_score
compara = pd.DataFrame(y_test)

compara['predictKNN'] = y_pred

compara.head(100)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtc = DecisionTreeClassifier()

modelo = dtc.fit(X_train,y_train)

y_train = modelo.predict(X_test)

#guarde o resultado da predição em uma nova coluna "predictAD"

import graphviz as gpz

dot_data = export_graphviz(dtc)

graph = gpz.Source(dot_data)

graph

