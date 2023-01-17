# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import spatial
%matplotlib inline
import  statistics
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importando dataset
data_iris= pd.read_csv("/kaggle/input/iris/Iris.csv",usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])

#Motando conjunto de teste e treino
def set_train_test(frac_train,data_iris):
    
    #selecionando aleatoriamente 50% da base para treino
    #replace=False não terá repetição de dados
    train_set= data_iris.sample(frac=frac_train, replace=False)
    
    
    #retirando sobreposição para o conjunto de teste
    combined = data_iris.append(train_set)
    test_set= combined[~combined.index.duplicated(keep=False)]
    
    
    return train_set,test_set

train_set,test_set=set_train_test(0.5,data_iris)
#conjunto de treinamento e teste
train_set,test_set
#motando a mastriz de distância euclidiana 
def montar_matriz_de_distancia(test_set,train_set):
    distance_matrix= spatial.distance.cdist(test_set.iloc[:,:4],
                                            train_set.iloc[:,:4], metric='euclidean')
    
    #coluna representa o indice do conjunto de treino e linha o indece do conjunto de teste
    return pd.DataFrame(distance_matrix,columns=train_set.index,index=test_set.index)

    
distance_matrix=montar_matriz_de_distancia(test_set,train_set)

distance_matrix
#selecionando 1_nn vizinhos mais próximo
def selecionar_k_vizinhos_proximos(quant_vizinho,distance_matrix):
    trans_matrix=distance_matrix.T
    lista=[]
    for k in distance_matrix.index:
        #selecionando o vizinho mais próximo
        ex_kNN=trans_matrix[k].nsmallest(n=quant_vizinho, keep='first')
        serie={'menor_peso':ex_kNN.values[0], 'index_menor_peso':ex_kNN.index[0]}
        lista.append(serie)
    return  pd.DataFrame(lista,index=[distance_matrix.index])

#classificar espécie do conjunto de teste utilizando 1_nn sem peso  
def classificar_especie_knn_sem_peso(lista_k_nn,test_set,train_set):
    
    #criando colunas
    lista_k_nn_copy=lista_k_nn.copy()
    lista_k_nn['Specie']=''
    lista_k_nn['classificação']=''
    
    #recuperando index da espécie classificada com menor peso
    lista_index=lista_k_nn_copy['index_menor_peso']
    
    for i in range(len(lista_k_nn_copy)):
        
        #recuperando a classe correta
        lista_k_nn.iloc[i:i+1,2:3]=[test_set.loc[lista_k_nn.index[i]]['Species']]
        
        #classificação
        lista_k_nn.iloc[i:i+1,3:]=[train_set.loc[lista_index.iloc[i]]['Species']]
        
    return lista_k_nn

lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
classificar_especie_knn_sem_peso(lista_k_nn,test_set,train_set)

#Calculando a taxa de acerto do classificador
def calcular_taxa_acerto_classificador(classif):
    ac_setosa=len(classif[(classif['classificação']=='Iris-setosa') &
                             (classif['Specie']=='Iris-setosa')])
    ac_versicolor=len(classif[(classif['classificação']=='Iris-versicolor') &
                             (classif['Specie']=='Iris-versicolor')])
    ac_virginica=len(classif[(classif['classificação']=='Iris-virginica') &
                             (classif['Specie']=='Iris-virginica')])
    
    return (ac_setosa+ac_versicolor+ac_virginica)/len(classif)*100

def holdout_aleatorio_com_repeticao(n):
    
    dict_taxa={}
    lista_acerto=[]
    for k in range(n):
        
        # conjunto de treino e teste
        train_set,test_set=set_train_test(0.5,data_iris)
        #matriz de distância
        distance_matrix=montar_matriz_de_distancia(test_set,train_set)
        #selecionando 1_nn vizinhos
        lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
        #classificando 
        classif=classificar_especie_knn_sem_peso(lista_k_nn,test_set,train_set)
        #calculando média de acerto do classificador
        taxa_acerto=calcular_taxa_acerto_classificador(classif)
        dict_taxa={'media_acerto_classificador': taxa_acerto}
        
        lista_acerto.append(dict_taxa)
    return pd.DataFrame(lista_acerto)
classif_holdout=holdout_aleatorio_com_repeticao(100)

classif_holdout
#Média da taxa de acerto
media_acerto_geral=classif_holdout['media_acerto_classificador'].mean()
#Máximo da taxa de acerto
maxima_taxa_acerto=classif_holdout['media_acerto_classificador'].max()
#Mínimo da taxa de acerto
minima_taxa_acerto=classif_holdout['media_acerto_classificador'].min()

#visualização
resultado = { 'Média da taxa de acerto':"%.2f" %  media_acerto_geral,
             'Máximo da taxa de acerto': "%.2f" %  maxima_taxa_acerto,
             'Mínimo da taxa de acerto': "%.2f" % minima_taxa_acerto}

resultado
classif_holdout.plot.hist(bins=10)
#calculando o desvio padrão
desvio_padrao=statistics.stdev(classif_holdout['media_acerto_classificador'])
#intervalo de confiança
["%.2f" %(media_acerto_geral-1.96*desvio_padrao),"%.2f" %(media_acerto_geral+1.96*desvio_padrao)]
