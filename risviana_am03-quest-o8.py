# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import spatial
from sklearn.metrics import recall_score
import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importando dataset
data_wine= pd.read_csv("/kaggle/input/wine-daset/wineDATA.CSV")

#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set,n):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[n:,2:14],
                                            test_train_set.iloc[:n,2:14], metric='euclidean')

    #coluna representa o indice do conjunto de treino e linha o indece do conjunto de teste
    return pd.DataFrame(distance_matrix,index=test_train_set.iloc[n:].index)
#selecionando 1_nn vizinhos mais próximo
def selecionar_k_vizinhos_proximos(quant_vizinho,distance_matrix):
    trans_matrix=distance_matrix.T
    lista=[]
    for k in distance_matrix.index:
        ex_kNN=trans_matrix[k].nsmallest(n=quant_vizinho, keep='first')
        serie={'menor_peso':ex_kNN.values[0], 'index_menor_peso':ex_kNN.index[0]}
        lista.append(serie)
    return  pd.DataFrame(lista,index=[distance_matrix.index])

#classificar espécie do conjunto de teste utilizando 1_nn sem peso  
def classificar_especie_knn_sem_peso(lista_k_nn,test_train_set):
    
    #criando colunas
    lista_k_nn_copy=lista_k_nn.copy()
    lista_k_nn['Class']=''
    lista_k_nn['classificação']=''
    lista_classif=[]
    lista_index=lista_k_nn_copy['index_menor_peso']
    for i in range(len(lista_k_nn_copy)):
        
        lista_k_nn.iloc[i:i+1,3:]=[test_train_set.iloc[lista_index.iloc[i]]['Class']]
        lista_k_nn.iloc[i:i+1,2:3]=[test_train_set.iloc[lista_k_nn.index[i]]['Class']]

    return lista_k_nn

#Motando conjunto de teste e treino
def Holdout_ateatorio(quant_treino,quant_test):
    
    train_set=data_wine.loc[random.sample(list(data_wine.index),quant_treino)]
    test_set=data_wine.loc[random.sample(list(data_wine.index),quant_test)]
    test_train_set=pd.concat([train_set, test_set]).reset_index()
    return test_train_set


quant_treino=int(len(data_wine)*90/100)
quant_test=int(len(data_wine)-quant_treino)
test_train_set=Holdout_ateatorio(quant_treino,quant_test)
distance_matrix=montar_matriz_de_distancia(test_train_set,quant_treino)
lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)

classif=classificar_especie_knn_sem_peso(lista_k_nn,test_train_set)
taxa_acerto=recall_score(classif['Class'].astype(int), classif['classificação'].astype(int), average='macro')
print ("TAXA DE ACERTO="+ str(taxa_acerto))
def holdout_aleatorio_com_repeticao(n):
    
    dict_taxa={}
    lista_acerto=[]
    for k in range(10):
        test_train_set=Holdout_ateatorio(quant_treino,quant_test)
        distance_matrix=montar_matriz_de_distancia(test_train_set,quant_treino)
        lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
        classif=classificar_especie_knn_sem_peso(lista_k_nn,test_train_set)
        media_acerto=recall_score(classif['Class'].astype(int), classif['classificação'].astype(int), average='macro')
        taxa_acerto1=len(classif[(classif['Class']==1) & (classif['classificação']==1)])
        taxa_acerto2=len(classif[(classif['Class']==2) & (classif['classificação']==2)])
        taxa_acerto3=len(classif[(classif['Class']==3) & (classif['classificação']==3)])
        dict_taxa={'taxa_acerto1':taxa_acerto1,'taxa_acerto2':taxa_acerto2,
                   'taxa_acerto3':taxa_acerto3,'media_acerto_classificador':media_acerto}
        lista_acerto.append(dict_taxa)
    return pd.DataFrame(lista_acerto)
holdout_aleatorio_com_repeticao(10)
#Motando conjunto de teste e treino
def holdout_estratificado(quant_treino,quant_test):
    
    train_set=data_wine.sample(n=quant_treino)
    test_set=data_wine.sample(n=quant_test)
    test_train_set=pd.concat([train_set, test_set]).reset_index()
    return test_train_set
def holdout_estratificado_com_repeticao(n):
    
    dict_taxa={}
    lista_acerto=[]
    for k in range(10):
        test_train_set=holdout_estratificado(quant_treino,quant_test)
        distance_matrix=montar_matriz_de_distancia(test_train_set,quant_treino)
        lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
        classif=classificar_especie_knn_sem_peso(lista_k_nn,test_train_set)
        media_acerto=recall_score(classif['Class'].astype(int), classif['classificação'].astype(int), average='macro')
        taxa_acerto1=len(classif[(classif['Class']==1) & (classif['classificação']==1)])
        taxa_acerto2=len(classif[(classif['Class']==2) & (classif['classificação']==2)])
        taxa_acerto3=len(classif[(classif['Class']==3) & (classif['classificação']==3)])
        dict_taxa={'taxa_acerto1':taxa_acerto1,'taxa_acerto2':taxa_acerto2,
                   'taxa_acerto3':taxa_acerto3,'media_acerto_classificador':media_acerto}
        lista_acerto.append(dict_taxa)
    return pd.DataFrame(lista_acerto)
holdout_estratificado_com_repeticao(10)
def separar_k_fold_subconjunto(n):
    lista_set=[]
    for k in range(n):
        k_fold=data_wine.sample(n=18)
        lista_set.append(k_fold)

    test_train_set=pd.concat(lista_set).reset_index()
    return test_train_set
def k_fold_cross_validation(n,test_train_set):
    
    dict_taxa={}
    lista_acerto=[]
    for k in range(10):
        distance_matrix=montar_matriz_de_distancia(test_train_set,162)
        lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
        classif=classificar_especie_knn_sem_peso(lista_k_nn,test_train_set)
        media_acerto=recall_score(classif['Class'].astype(int), classif['classificação'].astype(int), average='macro')
        taxa_acerto1=len(classif[(classif['Class']==1) & (classif['classificação']==1)])
        taxa_acerto2=len(classif[(classif['Class']==2) & (classif['classificação']==2)])
        taxa_acerto3=len(classif[(classif['Class']==3) & (classif['classificação']==3)])
        dict_taxa={'taxa_acerto1':taxa_acerto1,'taxa_acerto2':taxa_acerto2,
                   'taxa_acerto3':taxa_acerto3,'media_acerto_classificador':media_acerto}
        lista_acerto.append(dict_taxa)
    return pd.DataFrame(lista_acerto)
test_train_set=separar_k_fold_subconjunto(10)
fold_cross_validation=k_fold_cross_validation(10,test_train_set)
fold_cross_validation
#TAXA DE ACERTO DAS 10_FOLDS
print('Taxa_acerto_10_folds='+ str(fold_cross_validation['media_acerto_classificador'].sum()))
def separar_k_fold_subconjunto(tam):
    lista_set=[]
    test_set=data_wine.sample(n=tam)
    train_set=data_wine.sample(n=tam)
    test_train_set=pd.concat([test_set, train_set]).reset_index()
    return test_train_set

#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[:,2:14],
                                            test_train_set.iloc[:,2:14], metric='euclidean')

    #coluna representa o indice do conjunto de treino e linha o indece do conjunto de teste
    return pd.DataFrame(distance_matrix,index=test_train_set.iloc[:].index)
def k_fold_cross_validation(test_train_set):
    
    
    distance_matrix=montar_matriz_de_distancia(test_train_set)
    lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
    classif=classificar_especie_knn_sem_peso(lista_k_nn,test_train_set)
    taxa_acerto=recall_score(classif['Class'].astype(int), classif['classificação'].astype(int), average='macro')
        
    return taxa_acerto
test_train_set=separar_k_fold_subconjunto(178)
fold_cross_validation=k_fold_cross_validation(test_train_set)
print("taxa_acerto="+str(fold_cross_validation))