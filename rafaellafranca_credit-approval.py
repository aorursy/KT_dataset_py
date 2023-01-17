import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

df = pd.read_csv("/kaggle/input/credit-screening.data", sep=",",names=nomes, na_values= '?')

df.head()
df.columns
df.describe()
df.isnull().sum()
#preencher com a moda: A1, A4, A5, A6, A7  

cols=['A1', 'A4', 'A5', 'A7']

for i in cols:

    df[i]=df[i].fillna(df[i].mode)
df.isnull().sum()
#preencher com média: A2 e A14 

cols=['A2', 'A14']

for i in cols:

    df[i]=df[i].fillna(df[i].mean)

df.isnull().sum()
df.info()
cols=['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

for i in cols:

    df[i]=df[i].astype(str)
for i in cols:

    df[i]=df[i].astype('category')
df["A16"] = df["A16"].astype('category')

df["A16"].cat.codes
df = pd.get_dummies(df, columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
x = df.iloc[:,:]

y = df.iloc[:,:]

x.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

x_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

modelo= knn.fit(x_train,y_train)

y_pred = modelo.predict(x_test)

#guarde o resultado da predição em uma nova coluna do dataframe "predictKNN"
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtc = DecisionTreeClassifier()

modelo = dtc.fit(x_train,y_train)

y_pred = modelo.predict(x_test)

#guarde o resultado da predição em uma nova coluna "predictAD"

import graphviz as gpz

dot_data = export_graphviz(dtc)

graph = gpz.Source(dot_data)

graph