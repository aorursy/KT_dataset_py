import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
file_treino="../input/adult-pmr3508/train_data.csv"

file_test="../input/adult-pmr3508/test_data.csv"
base_treino=pd.read_csv(file_treino,

        names=

        ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

base_teste=pd.read_csv(file_test,

        names=

        ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", ],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
base_treino.head()
base_teste.head()
#retirar a primeira linha, com os titulos antigos

base_treino.drop(base_treino.index[0],inplace=True)

base_teste.drop(base_teste.index[0],inplace=True)
base_treino.shape
base_teste.shape
base_treino["Country"].value_counts()
base_treino["Age"].value_counts().plot(kind="bar")
base_teste["Age"].value_counts().plot(kind="bar")


base_treino["Sex"].value_counts().plot(kind="bar")
#Retirar linhas com dados faltantes

n_treino=base_treino.dropna()

N_teste=base_teste.dropna()
n_treino.shape
Xtreino = n_treino[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Ytreino = n_treino.Target

Xteste = base_teste[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

#Realizar a otimizacao do processo com o método de validacao cruzada

lista=[]

lista_score=[]

melhor=0 # melhor valor de k

med_max=0 # media das validacoes cruzadas

i=3

while i<=30: #vamos testar k de 3 ate no max 30

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)

    media=np.mean(scores) # media dos valores da validação cruzada

    if media>med_max: 

        media_max=media # armzena melhor media

        melhor=i # indicador de indice da melhor

    lista.append(i)

    lista_score.append(media)

    i+=1

plt.plot(lista,lista_score)
knn = KNeighborsClassifier(n_neighbors=melhor) #utiliza o melhor para o knn

knn.fit(Xtreino,Ytreino)
YtestePred = knn.predict(Xteste)
YtestePred
#criar arquivo de resultado

savepath = "predictions.csv"

prev = pd.DataFrame(YtestePred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev