# Importação de bibliotecas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Leitura da base de treino
treino = pd.read_csv("../input/train_data.csv")
# Dimensões da base
treino.shape
# Visão geral da base
treino
# Dados estatísticos da base
treino.describe()
# Remoção de linhas com dados faltantes
treino.dropna()
treino["native.country"].value_counts()
treino["sex"].value_counts().plot(kind="bar")
treino["age"].value_counts().plot(kind="bar")
treino["education"].value_counts().plot(kind="bar")
ntreino["occupation"].value_counts().plot(kind="bar")
Xtreino = treino.drop(columns=["income"])
Xtreino = Xtreino.apply(preprocessing.LabelEncoder().fit_transform)
Xtreino.head()
Ytreino = treino["income"]
Ytreino.head()
kNN = KNeighborsClassifier(n_neighbors=24)
LDA = LinearDiscriminantAnalysis()
LDA.fit(Xtreino, Ytreino)
DT = DecisionTreeClassifier()
DT.fit(Xtreino, Ytreino)
RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
RF.fit(Xtreino, Ytreino)
# kNN
score_kNN = cross_val_score(kNN, Xtreino, Ytreino, cv=10)
score_kNN
# LDA
score_LDA = cross_val_score(LDA, Xtreino, Ytreino, cv=10)
score_LDA
# Árvore de Decisão
score_DT = cross_val_score(DT, Xtreino, Ytreino, cv=10)
score_DT
# Random Forest
score_RF = cross_val_score(RF, Xtreino, Ytreino, cv=10)
score_RF
# Leitura da base de teste
Xteste = pd.read_csv("../input/test_data.csv")
Xteste = Xteste.apply(preprocessing.LabelEncoder().fit_transform)
Xteste.head()
Yteste = LDA.predict(Xtest)
Yteste
pred = pd.DataFrame(Xteste.Id)
pred["income"] = Yteste
pred.to_csv("prediction.csv", index=False)
