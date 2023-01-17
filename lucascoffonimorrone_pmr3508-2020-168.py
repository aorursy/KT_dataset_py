#Importando algumas bibliotecas importantes

import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing



%matplotlib inline
treino = pd.read_csv('../input/adult-pmr3508/train_data.csv', na_values = '?')



treino
treino.describe()
treino.groupby(['sex', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
treino.groupby(['relationship', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
treino.groupby(['age', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
treino.groupby(['education', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
treino.groupby(['occupation', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
treino['native.country'].value_counts()
treino_limpo = treino.dropna()

treino_limpo
treino_limpo.describe()
teste = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values = "?")

teste
teste_limpo = teste.dropna()

teste_limpo
numericos = ['age', 'education.num', 'capital.gain', 'capital.loss']

categoricos = ['sex', 'race', 'occupation', 'relationship', 'marital.status']

parametros = numericos + categoricos
treino_numerico = treino.fillna('?')

treino_numerico = pd.concat((treino_numerico[numericos], treino_numerico[categoricos].apply(preprocessing.LabelEncoder().fit_transform)), axis = 1)
X_treino = treino_numerico[parametros]

Y_treino = treino['income']
%%time



acuracia = []



for k in range(20, 40):

    knn = KNeighborsClassifier(k, p = 1)

    pontos = cross_val_score(knn, X_treino, Y_treino, cv = 10)

    acuracia.append(np.mean(pontos))



melhor_K = np.argmax(acuracia) + 15

print("Melhor Acuracia: {}, K = {}".format(max(acuracia), melhor_K))
knn = KNeighborsClassifier(melhor_K, p = 1)

knn = knn.fit(X_treino, Y_treino)
teste_numerico = teste.fillna('?')

teste_numerico = pd.concat((teste_numerico[numericos], teste_numerico[categoricos].apply(preprocessing.LabelEncoder().fit_transform)), axis = 1)
X_teste = teste_numerico[parametros]

Y_teste = knn.predict(X_teste)
id_posicao = pd.DataFrame({'Id' : list(range(len(Y_teste)))})

income = pd.DataFrame({'income' : Y_teste})

resultado = income

resultado
resultado.to_csv("submissao.csv", index = True, index_label = 'Id')