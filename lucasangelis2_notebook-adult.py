## Notebook Análise Database adult
# Imports para as bibliotecas necessárias
import pandas as pd
import sklearn
### SEÇAO 1: ANALISE DA BASE
#Leitura da Base completa
adultFullDB = pd.read_csv("../input/data-tarefa1/Adults_completa.txt",
                    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num","Martial Status","Occupation", "Relationship", "Race", "Sex","Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")
#import biblioteca de plot
import matplotlib.pyplot as plt
# Grau de Escolaridade
adultFullDB["Education"].value_counts().plot(kind="bar")
# Martial Status
adultFullDB["Martial Status"].value_counts().plot(kind="bar")
# Relationship
adultFullDB["Relationship"].value_counts().plot(kind="bar")
# Race
adultFullDB["Race"].value_counts().plot(kind="bar")
# Renda dos jovens com menos de 25 anos
data_rich = adultFullDB[(adultFullDB.Age < 25)]
data_rich["Target"].value_counts().plot(kind="bar")
# Renda da população Branca
data_white = adultFullDB[(adultFullDB.Race == "White")]
data_white["Target"].value_counts().plot(kind="bar")
# Renda da população em geral (excluindo brancos)
data_not_white = adultFullDB[(adultFullDB.Race != "White")]
data_not_white["Target"].value_counts().plot(kind="bar")
# Renda dos Not-in-family
data_not_f = adultFullDB[(adultFullDB.Relationship == "Not-in-family")]
data_not_f["Target"].value_counts().plot(kind="bar")
### SECAO 2: APLICACAO DO kNN
# DATABASE para treino
# Leitura da base de dados 
adultDB = pd.read_csv("../input/data-tarefa1/train_data.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")
#Visualização da base de dados para conferência
adultDB.head()
# Tamanho base de treino
adultDB.shape
# Retirando linhas com dados faltantes
nadultDB = adultDB.dropna()
#Exibir base "limpa"
n_lin = 20
nadultDB.iloc[0:n_lin,:]
#Selecionando apenas atributos numéricos para aplicação do kNN
Xadult = nadultDB[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
#Selecionando a coluna Target
Yadult = nadultDB.income
## BASE DE TESTE
# Lendo Base de Teste 
adultDBTest = pd.read_csv("../input/data-tarefa1/test_data.csv",
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
# Tamanho base teste
adultDBTest.shape
# Retirando linhas com valores faltantes
nadultDBTest = adultDBTest.dropna()
#Selecionando apenas atributos numéricos para aplicação do kNN
XtestAdult = nadultDBTest[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
## kNN
# Import biblioteca para utilizar classificador kNN
from sklearn.neighbors import KNeighborsClassifier
# Definição do classificador kNN
knn = KNeighborsClassifier(n_neighbors=30)
# Import bibliteca para realização de "Cross Validation"
from sklearn.model_selection import cross_val_score
# Procurar valor de cv que proporciona melhor media de scores
import numpy as np
values = np.array([10,20,30])
mean = 0;
aux = 0;
ind = 0;
for i in range(values.size):
    scores = cross_val_score(knn, Xadult, Yadult, cv = values[i])
    mean = np.mean(scores)
    if (mean > aux):
        aux = mean
        ind = i
print("melhor valor para cv eh",values[ind])
# Realização de "Cross Validation" melhor resultado de cv
scores = cross_val_score(knn, Xadult, Yadult, cv = 20)
# Media dos scores de cada bloco de dados
np.mean(scores)
# Realiza o fit no classificador
knn.fit(Xadult,Yadult)
# Predição dos valores com base no classificador
YtestPred = knn.predict(XtestAdult)
YtestPred
# OUTROS TESTES
##Incluinado "Marital Status" nas variáveis
#Convertendo valores nao numéricos 
from sklearn import preprocessing
numAdult = nadultDB.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nadultDBTest.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","marital.status","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = numAdult.income
XtestAdult = numTestAdult[["age","education.num","marital.status","capital.gain", "capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv = 20)
YtestPred = knn.predict(XtestAdult)
print("Scores: ",scores)
print("\n")
print("Media Scores: ",np.mean(scores))
print("\n")
print("Predicoes: ",YtestPred)
##Incluinado "Martial Status" e "Relationship" e k = 25
#Convertendo valores nao numéricos 
numAdult = nadultDB.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nadultDBTest.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","marital.status","relationship","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = numAdult.income
XtestAdult = numTestAdult[["age","education.num","marital.status","relationship","capital.gain", "capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv = 20)
YtestPred_best = knn.predict(XtestAdult)
print("Scores: ",scores)
print("\n")
print("Media Scores: ",np.mean(scores))
print("\n")
print("Predicoes: ",YtestPred_best)
##Incluinado "Martial Status", "Relationship" e "Sex"
#Convertendo valores nao numéricos 
numAdult = nadultDB.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nadultDBTest.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","marital.status","relationship","sex","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = numAdult.income
XtestAdult = numTestAdult[["age","education.num","marital.status","relationship","sex","capital.gain", "capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv = 20)
YtestPred = knn.predict(XtestAdult)
print("Scores: ",scores)
print("\n")
print("Media Scores: ",np.mean(scores))
print("\n")
print("Predicoes: ",YtestPred)