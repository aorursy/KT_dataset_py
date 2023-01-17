import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn as sk

import seaborn as sbn

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col = ['Id'], na_values = '?')
adult.head()
adult.shape
adult.describe()
sbn.catplot(x="income", y="hours.per.week", kind="boxen", data=adult);
sbn.catplot(x="income", y="education.num", kind="boxen", data=adult);
sbn.catplot(x="income", y="age", kind="boxen", data=adult);
sbn.catplot(x="income", y="fnlwgt", kind="boxen", data=adult);
sbn.catplot(x="income", y="capital.gain", kind="boxen", data=adult);
sbn.catplot(x="income", y="capital.loss", kind="boxen", data=adult);
adult01 = adult.copy()

le = LabelEncoder()

adult01['income'] = le.fit_transform(adult01['income'])
sbn.catplot(y="sex", x="income", kind="bar", data=adult01);
sbn.catplot(y="race", x="income", kind="bar", data=adult01);
sbn.catplot(y="education", x="income", kind="bar", data=adult01);
sbn.catplot(y="occupation", x="income", kind="bar", data=adult01);
adult01["occupation"].value_counts()
sbn.catplot(y="native.country", x="income", kind="bar", data=adult01);
ocorr = adult01["native.country"].value_counts()

ocorr
PorcAmer = (sum(ocorr) - ocorr[1])/sum(ocorr)*100

print("Porcentagem de Americaddo:", PorcAmer,"%")
adult.drop_duplicates(keep='first', inplace=True)
adult = adult.drop(['native.country','education', 'fnlwgt','marital.status' ], axis=1)
adult.head()
Ytreino = adult.pop('income')

Xtreino = adult
numericos = list(Xtreino.select_dtypes(include=[np.number]).columns.values)

print(numericos)
categoricos = list(Xtreino.select_dtypes(exclude=[np.number]).columns.values)

print(categoricos)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline



pipel_cat = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(drop='if_binary'))

])
from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer



pipel_num = Pipeline(steps = [

    ('imputer', KNNImputer(n_neighbors=10, weights="uniform")),

    ('scaler', StandardScaler())

])
from sklearn.compose import ColumnTransformer



# Cria o nosso Pré-Processador



# Cada pipeline está associada a suas respectivas colunas no dataset

preprocess = ColumnTransformer(transformers = [

    ('num', pipel_num, numericos),

    ('cat', pipel_cat, categoricos)

])
Xtreino = preprocess.fit_transform(Xtreino)
from sklearn.neighbors import KNeighborsClassifier



# Instancia nosso classificador

knn = KNeighborsClassifier(n_neighbors=41)
from sklearn.model_selection import cross_val_score



score = cross_val_score(knn, Xtreino, Ytreino, cv = 10, scoring="accuracy")

print("Score com cross validation:", score.mean())
knn = KNeighborsClassifier(n_neighbors=41)
knn.fit(Xtreino, Ytreino)
teste = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

Xteste = teste.drop(['native.country','education', 'fnlwgt','marital.status'], axis=1)
Xteste = preprocess.transform(Xteste)
predictions = knn.predict(Xteste)
submissao = pd.DataFrame()
submissao[0] = teste.index

submissao[1] = predictions

submissao.columns = ['Id','income']
submissao.to_csv('submission.csv',index = False)