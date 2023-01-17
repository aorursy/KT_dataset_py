# Todas importacoes



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from tqdm import tqdm
#abertura do arquivo com pandas

treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

                       sep=r'\s*,\s*',

                       engine='python',

                       na_values="?")
#dimensao do arquivo de treino

print('Dimensão do arquivo treino:', treino.shape)
#observando tabela

treino.head()
#limpando a base

treino = treino.dropna()



#vendo dimensões da base limpa

print('Dimensão do arquivo treino limpo:', treino.shape)
#abrindo base de teste

teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                       sep=r'\s*,\s*',

                       engine='python',

                       na_values="?")

#limpando base

teste.dropna()

#vendo dimensoes da nova base

print('Dimensão do arquivo teste:', teste.shape)
#observando tabela

teste.head()
X_treino = treino[['age','education.num','capital.gain','capital.loss','hours.per.week']]

Y_treino = treino['income']



#fazendo analise e econtrando o melhor classificador

k_max = 30

n_folds = 10



melhor_k = 1

melhor_acuracia = 0

lista_acuracia = []

for k in tqdm(range(1,k_max+1)):

    clf = KNeighborsClassifier(n_neighbors=k)

    acuracia = cross_val_score(clf, X_treino, Y_treino, cv = n_folds)

    lista_acuracia.append(acuracia.mean())



    if acuracia.mean() > melhor_acuracia:

        melhor_acuracia = acuracia.mean()

        melhor_k = k
#deixando resultados mais visuais

print("Valor de k = ", melhor_k)

print("Melhor acurácia encontrada = ", melhor_acuracia)