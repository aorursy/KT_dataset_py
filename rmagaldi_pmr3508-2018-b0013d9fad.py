#bliblioteca para facilitar as operações vetoriais e matriciais 
import numpy as np

#biblioteca para facilitar a organização do dataset
import pandas as pd

#blibliotecas de visualização gráfica
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
%matplotlib inline
import seaborn as sns

#biblioteca de machine learning
import sklearn
data_raw = pd.read_csv("../input/train_data.csv")
data_raw.sample(10)
data = data_raw.drop(axis=1, columns="Id")
data.head()
data.describe()
data.info()
data_X = data.drop(axis=1, columns="income")
data_Y = data.income
data_Y.sample(8)
white = data[data.race=="White"]
not_white = data[data.race!="White"]

male = data[data.sex=="Male"]
female = data[data.sex=="Female"]
data.income.value_counts().plot(kind="pie", radius=1.2, autopct='%1.1f%%')
male.income.value_counts().plot(kind="pie", radius=1.2, autopct='%1.1f%%')
female.income.value_counts().plot(kind="pie", radius=1.2, autopct='%1.1f%%')
white.income.value_counts().plot(kind="pie", radius=1.2, autopct='%1.1f%%')
not_white.income.value_counts().plot(kind="pie", radius=1.2, autopct='%1.1f%%')
data.race.value_counts().plot(kind="bar")
data["native.country"].value_counts().plot(kind="bar")
data_X_revised = data_X.drop(axis=1, columns=["workclass", "education", "marital.status", "occupation", "relationship", "native.country"])
data_X_revised
data_X_revised.sex.replace("Male", 1, inplace=True)
data_X_revised.sex.replace("Female", 0, inplace=True)

data_X_revised.race.replace("White", 1, inplace=True)
data_X_revised.race.replace(["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], 0, inplace=True)
data_X_revised.sample(5)
test_raw = pd.read_csv("../input/test_data.csv")
test = test_raw.drop(axis=1, columns="Id")
test_revised = test.drop(axis=1, columns=["workclass", "education", "marital.status", "occupation", "relationship", "native.country"])
test_revised.sex.replace("Male", 1, inplace=True)
test_revised.sex.replace("Female", 0, inplace=True)

test_revised.race.replace("White", 1, inplace=True)
test_revised.race.replace(["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], 0, inplace=True)
test_revised.sample(8)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# Obtendo o melhor k

lista_knn = []
for i in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, data_X_revised, data_Y, cv=10).mean()
    lista_knn.append(scores)
lista_knn
max(lista_knn)
# Implementando o resultado para o melhor k

def_knn = KNeighborsClassifier(n_neighbors=25)
def_knn.fit(data_X_revised, data_Y)
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
cross_val_score(dtc, data_X_revised, data_Y, cv=10)
dtc.fit(data_X_revised, data_Y)
from sklearn import svm
svmachine = svm.LinearSVC()
cross_val_score(svmachine, data_X_revised, data_Y, cv=5)
svmachine.fit(data_X_revised, data_Y)
from sklearn.ensemble import VotingClassifier
boosted_clf = VotingClassifier(estimators=[
...         ('knn', def_knn), ('decisionTree', dtc), ('svm', svmachine)], voting='hard')
boosted_clf.fit(data_X_revised, data_Y)
test_Y = boosted_clf.predict(test_revised)
test_Y
predict = pd.DataFrame(test_raw.Id)
predict["income"] = test_Y
predict
predict.to_csv("prediction_adult_boosted_v2.csv", index=False)