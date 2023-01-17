import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
#vamos guradar os endereços das bases de teste e de treino

file_treino="../input/adult-pmr3508/train_data.csv"

file_test="../input/adult-pmr3508/test_data.csv"
#criar as matrizes da base treino e teste

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

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
#olhar o começo de cada tabela

base_treino.head()
base_teste.head()
#retirar a primeira linha, com os titulos antigos

base_treino.drop(base_treino.index[0],inplace=True)

base_teste.drop(base_teste.index[0],inplace=True)
#verificar o head novamente

base_treino.head()
base_teste.head()
#verificando tamnaho de cada base

base_treino.shape
base_teste.shape
#agora, vamos verificar, visualmente, a distribuição de algumas de nossas variáveis
#distribuição das idades

base_treino["Age"].value_counts().plot(kind='bar')
#distribuição do tipo de trabalho

base_treino["Workclass"].value_counts().plot(kind='pie')
#distribuição do grau de escolaridade

base_treino["Education"].value_counts().plot(kind='pie')
base_treino["Education-Num"].value_counts().plot(kind='bar')
#distribuição de etnias

base_treino["Race"].value_counts().plot(kind='bar')
#distribuição de gênero

base_treino["Sex"].value_counts().plot(kind='bar')
#distribuição de nacionalidade

base_treino["Country"].value_counts()
#agora vamos eliminar as linhas que contenham dados faltantes

N_treino=base_treino.dropna()
#verificar o shape

N_treino.shape
#utilizaremos os dados numéricos para fazer o knn

Xtreino = N_treino[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Ytreino = N_treino.Target

Xteste =base_teste[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
#importar as bibliotecas pra rodar e testar o knn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



#agora vamos achar o valor de k para qual tenhamos o melhor resultado, ou seja, a maior media no resultado de validação cruzada

maximo=0 #variavel que guarda o k com melhor resultado

med_max=0 #variavel que guarda a maior media das validações cruzadas

i=3

while i<=60: #vamos testar k de 1 ate no max 60

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)

    med=np.mean(scores) #media dos valores da validação cruzada

    if med>med_max: #se essa média for maior que a maior já registrada, o melhor k muda

        med_max=med

        maximo=i

    i+=3
maximo
med_max
knn = KNeighborsClassifier(n_neighbors=maximo) #definir k como sendo o que assume a maior media da validação cruzada

knn.fit(Xtreino,Ytreino)
Yteste_previsao = knn.predict(Xteste) #aplicando o knn na base teste
Yteste_previsao
#criar arquivo de resultado

savepath = "results.csv"

prev = pd.DataFrame(Yteste_previsao, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev