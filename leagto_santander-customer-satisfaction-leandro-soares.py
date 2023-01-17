# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.



# Carrega a matriz de treinamento para um dataset.

matriz_treino_original = pd.read_csv(filepath_or_buffer = "../input/train.csv", index_col = 0)

matriz_teste_original = pd.read_csv(filepath_or_buffer = "../input/test.csv", index_col = 0)

print("At first: " + str(matriz_treino_original.shape))

print ("Satisfied Customers: " + str(matriz_treino_original[matriz_treino_original.TARGET == 0].shape))

print ("Unhappy Customers: " + str(matriz_treino_original[matriz_treino_original.TARGET == 1].shape))



# eliminaVarianciaZero removes from the dataset columns whose variance of values

# of rows is zero.

def eliminaVarianciaZero (matriz):

    # Para cada coluna, calcula-se a variância.

    variancia = matriz.var(axis = 0, numeric_only = True)

    matriz_sem_variancia_zero = matriz.loc[:, variancia.loc[variancia != 0].index]

    return matriz_sem_variancia_zero



matriz_treino = eliminaVarianciaZero(matriz_treino_original)

print("After eliminating zero variance: " + str(matriz_treino.shape))

# eliminaPercentil10IgualPercentil95 removes from the dataset the columns whose

# 10th percentile and 95th percentile are equal to each other, considering two

# datasets generated from the original array, one with the rows of

# TARGET equal to zero and one with the rows of TARGET equal to 1.

def eliminaPercentil10IgualPercentil95 (matriz):

    matriz_pos = matriz[matriz.TARGET == 0].describe(percentiles = [0.1, 0.95])

    matriz_neg = matriz[matriz.TARGET == 1].describe(percentiles = [0.1, 0.95])

    colunas = matriz.columns.drop('TARGET')

    colunas_manter = matriz.columns.where(matriz.columns == 'TARGET').dropna()



    for coluna in colunas:

        p10_pos = matriz_pos.loc['10%', coluna]

        p95_pos = matriz_pos.loc['95%', coluna]



        p10_neg = matriz_neg.loc['10%', coluna]

        p95_neg = matriz_neg.loc['95%', coluna]



        if not(p10_pos == p95_pos == p10_neg == p95_neg):

            colunas_manter = colunas_manter.insert(colunas_manter.size - 1, coluna)



    matriz_retorno = matriz.loc[:, colunas_manter]

    return matriz_retorno



matriz_treino = eliminaPercentil10IgualPercentil95(matriz_treino)

print("After eliminating 10 and 95 percentile equals: " + str(matriz_treino.shape))

# eliminaLinhasDuplicadas deletes duplicate rows in two steps:

# First it eliminates duplicate lines, considering TARGET, keeping only

# one of them.

# After this, delete all duplicate rows, except for TARGET.

def eliminaLinhasDuplicadas (matriz):

    matriz_retorno = matriz.drop_duplicates()

    matriz_retorno = matriz_retorno.drop_duplicates(subset = matriz_retorno.columns.drop('TARGET'), keep = False)

    return matriz_retorno



matriz_treino = eliminaLinhasDuplicadas(matriz_treino)

print("After deleting duplicate rows: " + str(matriz_treino.shape))

print ("Satisfied Customers: " + str(matriz_treino[matriz_treino.TARGET == 0].shape))

print ("Unhappy Customers: " + str(matriz_treino[matriz_treino.TARGET == 1].shape))

# Set up three datasets for training and validation of the classifier, the

# Following way:

# - The dataset is broken in two, depending on the classification of the sample.

# - Randomly select 15% of samples from each of the two sets

# To compose the validation set.

# - The remaining samples from the set with data classified as "Not satisfied"

# Composes two training sets, one balanced and one unbalanced.

# - The remaining samples from the set with the data classified as "Satisfied"

# Compose the unbalanced training set.

# - Randomly select the remaining samples from the set with the data

# Classified as "Satisfied" the number of remaining samples of the set with

# The data classified as "Not satisfied" to compose the set of

# Balanced training.



import random as rnd



ids_satisfeito = matriz_treino[matriz_treino.TARGET == 0].index

ids_nao_satisfeito = matriz_treino[matriz_treino.TARGET == 1].index



quantidade_amostras_validacao = round(0.15 * ids_satisfeito.size)

lista_id_aleatoria = rnd.sample(list(ids_satisfeito), quantidade_amostras_validacao)

ids_validacao = pd.Index(lista_id_aleatoria)

ids_satisfeito = ids_satisfeito.drop(lista_id_aleatoria)



quantidade_amostras_validacao = round(0.15 * ids_nao_satisfeito.size)

lista_id_aleatoria = rnd.sample(list(ids_nao_satisfeito), quantidade_amostras_validacao)

ids_validacao = ids_validacao.append(pd.Index(lista_id_aleatoria))



ids_treino_nao_balanceado = matriz_treino.index.drop(ids_validacao)



ids_treino_balanceado = ids_nao_satisfeito.drop(lista_id_aleatoria)

lista_id_aleatoria = rnd.sample(list(ids_satisfeito), ids_treino_balanceado.size)

ids_treino_balanceado = ids_treino_balanceado.append(pd.Index(lista_id_aleatoria))



matriz_validacao = matriz_treino.loc[ids_validacao, :]

matriz_treino_nao_balanceado = matriz_treino.loc[ids_treino_nao_balanceado, :]

matriz_treino_balanceado = matriz_treino.loc[ids_treino_balanceado, :]



print("Validation dataset: " + str(matriz_validacao.shape))

print("Unbalanced training dataset: " + str(matriz_treino_nao_balanceado.shape))

print("Balanced training dataset: " + str(matriz_treino_balanceado.shape))

# It creates classifiers, using K-Nearest Neighbor with k equal to 1 and k equals 5,

# with balanced and unbalanced data, and measures the accuracy of each of these

# models, considering the validation data.

import sklearn.neighbors as sknb

import sklearn.metrics as skmt



def validaClassificador(matriz_treinamento, matriz_validacao, numero_vizinhos):

    matriz_treino_atributos = matriz_treinamento.loc[:, matriz_treinamento.columns.drop(['TARGET'])]

    matriz_treino_alvo = matriz_treinamento.loc[:, 'TARGET']



    matriz_validacao_atributos = matriz_validacao.loc[:, matriz_validacao.columns.drop(['TARGET'])]

    matriz_validacao_alvo = matriz_validacao.loc[:, 'TARGET']



    classificador = sknb.KNeighborsClassifier(n_neighbors = numero_vizinhos)

    classificador.fit(matriz_treino_atributos, matriz_treino_alvo)

    

    predicao = pd.DataFrame(classificador.predict(matriz_validacao_atributos))



    acuracia = skmt.accuracy_score(matriz_validacao_alvo, predicao)



    return acuracia



acuracia_desbal_1 = validaClassificador(matriz_treino_nao_balanceado, matriz_validacao, 1)

print("Accuracy unbalanced model 1 neighbor: " + str(acuracia_desbal_1))



acuracia_desbal_5 = validaClassificador(matriz_treino_nao_balanceado, matriz_validacao, 5)

print("Accuracy unbalanced model 5 neighbors: " + str(acuracia_desbal_5))



acuracia_bal_1 = validaClassificador(matriz_treino_balanceado, matriz_validacao, 1)

print("Accuracy balanced model 1 neighbor: " + str(acuracia_bal_1))



acuracia_bal_5 = validaClassificador(matriz_treino_balanceado, matriz_validacao, 5)

print("Accuracy balanced model 5 neighbors: " + str(acuracia_bal_5))
