#Importar bibliotecas a serem utilizadas inicialmente

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#Leitura dos dados a serem trabalhados (Data frame: Adult para treino e teste)

Atrain = pd.read_csv("../input/adult-pmr3508/train_data.csv", header = 0, sep = ',', engine='python', na_values="?")

Atest = pd.read_csv("../input/adult-pmr3508/test_data.csv", header = 0, sep = ',', engine='python', na_values="?") 
#Visualizando os primeiros dados do data frame

Atrain.head()
#Visualizando os últimos dados do data frame

Atrain.tail()
#É notável que as variáveis são de dois tipos, inicialmente, categóricas e numéricas. Vamos olhar um pouco melhor estas variáveis.

#Primeiro as categóricas.

Atrain['workclass'].value_counts().plot(kind='bar')
Atrain['education'].value_counts().plot(kind='bar')
Atrain['marital.status'].value_counts().plot(kind='bar')
Atrain['occupation'].value_counts().plot(kind='bar')
Atrain['relationship'].value_counts().plot(kind='bar')
Atrain['race'].value_counts().plot(kind='bar')
Atrain['native.country'].value_counts().plot(kind='bar')
Atrain['income'].value_counts().plot(kind='bar')
#Agora as numéricas.

Atrain['age'].value_counts()
Atrain['education.num'].value_counts()
Atrain['fnlwgt'].value_counts()
Atrain['capital.gain'].value_counts()
Atrain['capital.loss'].value_counts()
Atrain['hours.per.week'].value_counts()
#Realizando algumas inferências com a variável que desejamos classificar:

tabela1 = pd.pivot_table(data = Atrain, values = 'Id', index = 'sex', columns = 'income', aggfunc = 'count')
print(tabela1)

tabela1.plot(kind='bar')
tabela2 = pd.pivot_table(data = Atrain, values = 'Id', index = 'education', columns = 'income', aggfunc = 'count')
print(tabela2)

tabela2.plot(kind='bar')
tabela3 = pd.pivot_table(data = Atrain, values = 'Id', index = 'occupation', columns = 'income', aggfunc = 'count')
print(tabela3)

tabela3.plot(kind='bar')
#Quantidade de linhas e colunas no data frame

Atrain.shape
#Análise detalhada(estatística) do conjunto de treino 

Atrain.describe()
#Retirando váriavel não importante, para cumprir nosso objetivo.

Atrain = Atrain.drop(columns=['fnlwgt'])
#Contagem de valores faltantes em cada uma das variáveis do conjunto de treino.

np.sum(Atrain.isna())
#Analisar as variáveis com valores faltantes

Atrain['workclass'].describe()
Atrain['occupation'].describe()
Atrain['native.country'].describe()
#Tratando os valores faltantes.

Atrain['workclass'] = Atrain['workclass'].fillna('Private')
Atrain['occupation'] = Atrain['workclass'].fillna('Private')
Atrain['native.country'] = Atrain['workclass'].fillna('Private')
#Verificando se funcionou o tratamento dos valores faltantes.

np.sum(Atrain.isna())
#Retirar a váriavel não importante do conjunto de teste.

Atest = Atest.drop(columns=['fnlwgt'])
np.sum(Atest.isna())
Atest['workclass'].describe()
Atest['occupation'].describe()
Atest['native.country'].describe()
#Tratando os valores faltantes

Atest['workclass'] = Atest['workclass'].fillna('Private')
Atest['occupation'] = Atest['workclass'].fillna('Private')
Atest['native.country'] = Atest['workclass'].fillna('Private')
#Da mesma forma verificamos se funcionou o tratamento.

np.sum(Atest.isna())
#Importar a biblioteca sklearn, que possui um pacote para pré-processamento dos dados, que seria está etapa sendo trabalhada.

from sklearn import preprocessing
#Variáveis não numéricas para numéricas.

numAtrain = Atrain.apply(preprocessing.LabelEncoder().fit_transform)

numAtest = Atest.apply(preprocessing.LabelEncoder().fit_transform)
#Detalhamento das variáveis númericas e esparsas (antes).

numAtrain[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]].describe()
#Variáveis Númericas

scaler = preprocessing.StandardScaler().fit(numAtrain[["age"]])
numAtrain["age"] = scaler.transform(numAtrain[["age"]])
scaler = preprocessing.StandardScaler().fit(numAtrain[["education.num"]])
numAtrain["education.num"] = scaler.transform(numAtrain[["education.num"]])
scaler = preprocessing.StandardScaler().fit(numAtrain[["hours.per.week"]])
numAtrain["hours.per.week"] = scaler.transform(numAtrain[["hours.per.week"]])
#Variáveis esparsas

scaler = preprocessing.RobustScaler().fit(numAtrain[["capital.gain"]])
numAtrain["capital.gain"] = scaler.transform(numAtrain[["capital.gain"]])
scaler = preprocessing.RobustScaler().fit(numAtrain[["capital.loss"]])
numAtrain["capital.loss"] = scaler.transform(numAtrain[["capital.loss"]])
#Detalhamento das variáveis númericas e esparsas (depois).

numAtrain[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]].describe()
#Conjunto de teste

#Detalhamento váriaveis numéricas e esparsas (antes)
numAtest[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]].describe()
#Variáveis Númericas

scaler = preprocessing.StandardScaler().fit(numAtest[["age"]])
numAtest["age"] = scaler.transform(numAtest[["age"]])
scaler = preprocessing.StandardScaler().fit(numAtest[["education.num"]])
numAtest["education.num"] = scaler.transform(numAtest[["education.num"]])
scaler = preprocessing.StandardScaler().fit(numAtest[["hours.per.week"]])
numAtest["hours.per.week"] = scaler.transform(numAtest[["hours.per.week"]])
#Variáveis esparsas

scaler = preprocessing.RobustScaler().fit(numAtest[["capital.gain"]])
numAtest["capital.gain"] = scaler.transform(numAtest[["capital.gain"]])
scaler = preprocessing.RobustScaler().fit(numAtest[["capital.loss"]])
numAtest["capital.loss"] = scaler.transform(numAtest[["capital.loss"]])
#Detalhamento das variáveis númericas e esparsas (depois).

numAtest[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]].describe()
#Importar as bibliotecas a serem utilizadas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#Os conjuntos organizados nas demais variáveis (XAtrain) e a variável que queremos classificar (YAtrain)

XAtrain = numAtrain[['age', 'workclass', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race','sex','capital.gain', 'capital.loss','hours.per.week','native.country']]

YAtrain = numAtrain.income
#Escolhi verificar 30 vizinhos próximos

final_score = []
melhorScore = 0
melhork = 0


for k in range(1,31):
    
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, XAtrain, YAtrain) 
    
    mScore = np.mean(scores)
    
    final_score.append(mScore)
    
    print('K atual: ', k)
    print('Score: ', mScore)
    
    if mScore > melhorScore:
        
        melhorScore = mScore
        melhorK = k
    
   
    print('Melhor score: ', melhorScore)
    print('Melhor K: ', melhorK)
    print('')
    
#Visualização dos valores de K obtidos

plt.scatter(range(1,31), final_score, color = 'red')
knn = KNeighborsClassifier(n_neighbors = 29)

knn.fit(XAtrain, YAtrain)
XAtest = numAtest[['age', 'workclass', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race','sex','capital.gain', 'capital.loss','hours.per.week','native.country']]
knn = KNeighborsClassifier(n_neighbors = 29)
knn.fit(XAtrain,YAtrain)
predicao = knn.predict(XAtest)
print(predicao)
#Predição resultou com 'income' numérico, então precisa-se transformar de volta para não numérica, afim de ter um melhor entendimento e visualização da predição.
predicao_final = []

for i in range(len(predicao)):
    if (predicao[i] == 0):
        predicao_final.append('<=50K')
    else:
        predicao_final.append('>50K')
#Por fim cria-se um data frame para a predição e o converte para um arquivo csv a ser submetido.


Predicao_DF = pd.DataFrame(predicao_final, columns = ['income'])

Predicao_DF.to_csv("submission.csv", index = True, index_label = 'Id')