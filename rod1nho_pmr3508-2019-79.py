import pandas as pd

import matplotlib.pyplot as plt

import sklearn

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", skipinitialspace = True, na_values = "?")

adult.set_index('Id',inplace=True)

adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 

                 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']

treino = adult.dropna()

treino
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv")

testAdult.set_index('Id',inplace=True)

teste = testAdult.dropna()

teste.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 

                 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']

teste.shape
adult["sex"].value_counts().plot(kind="bar")
adult["race"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["workclass"].value_counts().plot(kind="bar")
adult["age"].value_counts().plot(kind="bar")
nTreino = treino.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

nTreino = nTreino.join(treino.income)

nTreino.head()
nTeste = teste.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

nTeste.head()
xTreino = nTreino[["workclass", "education", "occupation", "race", "sex"]]

yTreino = treino.income



xTeste = nTeste[["workclass", "education", "occupation", "race", "sex"]]
knn = KNeighborsClassifier(n_neighbors=30)

xval = cross_val_score(knn, xTreino, yTreino, cv=10)

xval
knn = KNeighborsClassifier(n_neighbors=15)

xval = cross_val_score(knn, xTreino, yTreino, cv=10)

xval
xTreino = nTreino[["age", "workclass", "occupation", "sex", "capital.gain", "capital.loss"]]

yTreino = treino.income



xTeste = nTeste[["age", "workclass", "occupation", "sex", "capital.gain", "capital.loss"]]
knn = KNeighborsClassifier(n_neighbors=30)

xval = cross_val_score(knn, xTreino, yTreino, cv=10)

xval
knn = KNeighborsClassifier(n_neighbors=30)

xval = cross_val_score(knn, xTreino, yTreino, cv=10)

accuracy = xval.mean()

accuracy
knn = KNeighborsClassifier(n_neighbors=15)

xval = cross_val_score(knn, xTreino, yTreino, cv=10)

accuracy = xval.mean()

accuracy
knn = KNeighborsClassifier(n_neighbors=5)

xval = cross_val_score(knn, xTreino, yTreino, cv=10)

accuracy = xval.mean()

accuracy
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(xTreino, yTreino)

ytPred = knn.predict(xTeste)
id_index = pd.DataFrame({'Id' : list(range(len(ytPred)))})

income = pd.DataFrame({'income' : ytPred})

result = id_index.join(income)

result.to_csv("resposta.csv", index = False)

result