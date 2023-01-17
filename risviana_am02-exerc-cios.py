# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import spatial
import collections
import itertools
import matplotlib
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importando dataset
data = pd.read_csv("/kaggle/input/iris/Iris.csv",usecols=['SepalLengthCm','SepalWidthCm',
                                                          'PetalLengthCm','PetalWidthCm','Species'])
data.head()

def montar_conjunto_teste_treino1():
    #selecionando três exemplos aleatórios de cada espécie
    random_setosa = data[data['Species']=='Iris-setosa'].sample(n=3).copy()
    random_versicolor = data[data['Species']=='Iris-versicolor'].sample(n=3).copy()
    random_virginica = data[data['Species']=='Iris-virginica'].sample(n=3).copy()


    frame=[random_setosa.iloc[1:],random_versicolor.iloc[1:],random_virginica.iloc[1:],
                   random_setosa.iloc[0:1],random_versicolor.iloc[0:1],random_virginica.iloc[0:1]]

    test_train_set=pd.concat(frame).reset_index()

    test_train_set.iloc[6:,5:]='?'
    return test_train_set

test_train_set=montar_conjunto_teste_treino1()
#test_train_set

#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[6:,1:4],
                                            test_train_set.iloc[:6,1:4], metric='euclidean')

    return pd.DataFrame(distance_matrix,index=['EX6', 'EX7', 'EX8'])


distance_matrix=montar_matriz_de_distancia(test_train_set)
#distance_matrix

#CÓDIGO REUTIZIADO ABAIXO
#selecionando k vizinhos mais próximo
def selecionar_k_vizinhos_proximos(quant_vizinho,distance_matrix):
    trans_matrix=distance_matrix.T
    lista=[]
    
    for k in distance_matrix.index:
        ex_kNN=trans_matrix[k].nsmallest(n=quant_vizinho, keep='first')
        lista.append(ex_kNN)
        
    return  pd.DataFrame(lista)


#classificar espécie do conjunto de teste utilizando knn com peso  
def classificar_especie_knn_sem_peso(lista_k_nn):
    
    #criando colunas
    lista_k_nn_copy=lista_k_nn.copy()
    lista_k_nn['Quant_setosa']=0
    lista_k_nn['Quant_versicolor']=0
    lista_k_nn['Quant_virginica']=0
    lista_k_nn['classificação']=''
    
    for i in range(len(lista_k_nn_copy)):
        
        #eleminando colunas vazias
        lista=lista_k_nn_copy.iloc[i].dropna()
        lista_classif=[]
        tam=len(lista_k_nn.iloc[i])-4
        for k in range(len(lista)):
            
            classif=[test_train_set.iloc[lista.index[k]]['Species']]
            lista_classif.append(classif)
            
        lista_classif=list(itertools.chain.from_iterable(lista_classif))
        hash_classif=collections.Counter(lista_classif)
        lista_k_nn.iloc[i:i+1,tam:tam+1]=hash_classif['Iris-setosa']
        lista_k_nn.iloc[i:i+1,tam+1:tam+2]=hash_classif['Iris-versicolor']
        lista_k_nn.iloc[i:i+1,tam+2:tam+3]=hash_classif['Iris-virginica']
        lista_k_nn.iloc[i:i+1,tam+3:]=max(hash_classif, key=hash_classif.get)
    
    return lista_k_nn
def classificar_especie_knn_com_peso(lista_k_nn):
     #criando colunas
    lista_k_nn_copy=lista_k_nn.copy()
    lista_k_nn['peso_setosa']=0
    lista_k_nn['peso_versicolor']=0
    lista_k_nn['peso_virginica']=0
    lista_k_nn['classificação']=''
    
    for i in range(len(lista_k_nn_copy)):
       
        #eleminando colunas vazias
        lista=lista_k_nn_copy.iloc[i].dropna()
        lista_classif={}
        tam=len(lista_k_nn.iloc[i])-4
        for k in range(len(lista)):
            
            classif=[test_train_set.iloc[lista.index[k]]['Species']]
            if (lista.iloc[k]!=0):
                if(classif[0]=='Iris-versicolor'):
                    lista_k_nn.iloc[i:i+1,tam+1:tam+2]+=(1/lista.iloc[k])
                    lista_classif[classif[0]]=lista_k_nn.iloc[i]['peso_versicolor']
                elif(classif[0]=='Iris-setosa'):
                    lista_k_nn.iloc[i:i+1,tam:tam+1]+=(1/lista.iloc[k])
                    lista_classif[classif[0]]=lista_k_nn.iloc[i]['peso_setosa']
                elif(classif[0]=='Iris-virginica'):
                    lista_k_nn.iloc[i:i+1,tam+2:tam+3]+=(1/lista.iloc[k])
                    lista_classif[classif[0]]=lista_k_nn.iloc[i]['peso_virginica']
        lista_k_nn.iloc[i:i+1,tam+3:]=max(lista_classif, key=lista_classif.get)
                                      
    return lista_k_nn

#Classificação utilizando 1-NN com peso
     
lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
#classificar_especie_knn_com_peso(lista_k_nn)

#Classificação utilizando 3-NN com peso

lista_k_nn=selecionar_k_vizinhos_proximos(3,distance_matrix)
#classificar_especie_knn_com_peso(lista_k_nn)
#Classificação utilizando 3-NN sem peso

lista_k_nn=selecionar_k_vizinhos_proximos(3,distance_matrix)
tabela_classificação=classificar_especie_knn_sem_peso(lista_k_nn)
#tabela_classificação
        
#Diagrama de dispersão
pd.options.plotting.backend = "plotly"
np.random.seed(1)

fig = data.plot.scatter(data['SepalLengthCm'],data['Species'])
#fig.show()

#Classificador pelo vizinho mais próximo
def montar_conjunto_teste_treino2():
    
    frame=[data.iloc[10:50],data.iloc[60:100],data.iloc[110:],
           data.iloc[0:10],data.iloc[50:60],data.iloc[100:110]]
    test_train_set=pd.concat(frame).reset_index()
    return test_train_set

test_train_set=montar_conjunto_teste_treino2()

#test_train_set.head()




#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set,metrica,p):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[120:,1:5],
                                            test_train_set.iloc[:120,1:5], metric=metrica,p=p)

    return pd.DataFrame(distance_matrix)


distance_matrix=montar_matriz_de_distancia(test_train_set,'euclidean',1)
#distance_matrix.head()

#selecionando os vizinhos mais próximos e calssificando conjunto de teste
lista=selecionar_k_vizinhos_proximos(1,distance_matrix)
teste=test_train_set.iloc[120:,5:]
df=classificar_especie_knn_sem_peso(lista)
df.iloc[:,23:].head(10)
classsif=df['classificação']
teste=teste['Species']
acc=accuracy_score(teste,classsif)*100
print('Taxa de acerto com a métrica euclidean:',acc)
#p=1
distance_matrix=montar_matriz_de_distancia(test_train_set,'minkowski',1)
lista=selecionar_k_vizinhos_proximos(1,distance_matrix)
df=classificar_especie_knn_sem_peso(lista)
df.iloc[:,22:].head(10)


classsif=df['classificação']
teste=test_train_set.iloc[120:,5:]
teste=teste['Species']
acc=accuracy_score(teste,classsif)*100
print('Taxa de acerto com a métrica minkowski e peso 1:',acc)
#p=2
distance_matrix=montar_matriz_de_distancia(test_train_set,'minkowski',2)
lista=selecionar_k_vizinhos_proximos(1,distance_matrix)
df=classificar_especie_knn_sem_peso(lista)
df.iloc[:,23:].head(10)
classsif=df['classificação']
teste=test_train_set.iloc[120:,5:]
teste=teste['Species']
acc=accuracy_score(teste,classsif)*100
print('Taxa de acerto com a métrica minkowski e peso 2:',acc)
#p=4
distance_matrix=montar_matriz_de_distancia(test_train_set,'minkowski',4)
lista=selecionar_k_vizinhos_proximos(1,distance_matrix)
df=classificar_especie_knn_sem_peso(lista)
df.iloc[:,23:].head(10)
classsif=df['classificação']
teste=test_train_set.iloc[120:,5:]
teste=teste['Species']
acc=accuracy_score(teste,classsif)*100
print('Taxa de acerto com a métrica minkowski e peso 4:',acc)
#Classificador pelo vizinho mais próximo
def montar_conjunto_teste_treino2():
    
    frame=[data.iloc[25:50],data.iloc[75:100],data.iloc[125:],data.iloc[0:25],data.iloc[50:75],data.iloc[100:125]]
    test_train_set=pd.concat(frame).reset_index()
    
    return test_train_set

test_train_set=montar_conjunto_teste_treino2()

#test_train_set

#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[75:,1:5],
                                            test_train_set.iloc[:75,1:5], metric='euclidean')

    return pd.DataFrame(distance_matrix,index=test_train_set.iloc[75:150].index)


distance_matrix=montar_matriz_de_distancia(test_train_set)
#distance_matrix
#selecionando os 7_nn vizinhos mais próximos sem peso e calssificando conjunto de teste
lista=selecionar_k_vizinhos_proximos(7,distance_matrix)
teste=test_train_set.iloc[75:,5:]
df=classificar_especie_knn_sem_peso(lista)
df.iloc[:,74:].head(10)
classsif=df['classificação']
teste=teste['Species']
acc=accuracy_score(teste,classsif)*100
print('Taxa de acerto 7_nn vizinhos mais próximos sem peso:',acc)
#selecionando os 7_nn vizinhos mais próximos com peso e calssificando conjunto de teste
lista=selecionar_k_vizinhos_proximos(7,distance_matrix)
teste=test_train_set.iloc[75:,5:]
df=classificar_especie_knn_com_peso(lista)
df.iloc[:,74:].head(10)
classsif=df['classificação']
teste=teste['Species']
acc=accuracy_score(teste,classsif)*100
print('Taxa de acerto 7_nn vizinhos mais próximos com peso:',acc)
x_test=test_train_set.iloc[75:,1:5]
y_test=teste=test_train_set.iloc[75:,5:6]                                          
x_train=test_train_set.iloc[:75,1:5]
y_train=test_train_set.iloc[:75,5:6]
#selecionando os 7_nn vizinhos mais próximos sem peso e calssificando conjunto de teste
neigh = KNeighborsClassifier(n_neighbors=7,metric='euclidean')
knn=neigh.fit(x_train, y_train['Species'])
y_pred=knn.predict(x_test)
acc=accuracy_score(y_test['Species'],y_pred)*100
print('Taxa de acerto 7_nn vizinhos mais próximos sem peso da blibioteca do sklearn:',acc)

#selecionando os 7_nn vizinhos mais próximos com peso e calssificando conjunto de teste
neigh = KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights='distance')
knn=neigh.fit(x_train, y_train['Species'])
y_pred=knn.predict(x_test)
acc=accuracy_score(y_test['Species'],y_pred)*100
print('Taxa de acerto 7_nn vizinhos mais próximos com peso da blibioteca do sklearn:',acc)
#importando dataset
wine_data = pd.read_csv("/kaggle/input/wine-daset/wineDATA.CSV")
#wine_data.iloc[89:]

#Motando conjunto de teste e treino
def montar_conjunto_teste_treino3():
    
    frame=[wine_data.iloc[0:89],wine_data.iloc[89:]]
    test_train_set=pd.concat(frame)
    test_train_set.iloc[89:,0:1]='?'
    return test_train_set

test_train_set=montar_conjunto_teste_treino3()

#test_train_set.head()
#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[89:,1:],
                                            test_train_set.iloc[:89,1:], metric='euclidean')

    return pd.DataFrame(distance_matrix,index=test_train_set.iloc[89:187].index)


distance_matrix=montar_matriz_de_distancia(test_train_set)
#distance_matrix.head()
#Avaliando vários valores de k (sem) peso e calssificando conjunto de teste

def classificar_vinho_knn_sem_peso(lista_wine,test_train_set):
    #criando colunas
    lista_wine_copy=lista_wine.copy()
    lista_wine['Quant_class1']=0
    lista_wine['Quant_class2']=0
    lista_wine['Quant_class3']=0
    lista_wine['classificação']=''
    
    for i in range(len(lista_wine_copy)):
        
        #eleminando colunas vazias
        lista=lista_wine_copy.iloc[i].dropna()
        lista_classif=[]
        tam=len(lista_wine.iloc[i])-4
        for k in range(len(lista)):
            
            classif=[test_train_set.iloc[lista.index[k]]['Class']]
            lista_classif.append(classif)
        lista_classif=list(itertools.chain.from_iterable(lista_classif))
        hash_classif=collections.Counter(lista_classif)
        lista_wine.iloc[i:i+1,tam:tam+1]=hash_classif[1]
        lista_wine.iloc[i:i+1,tam+1:tam+2]=hash_classif[2]
        lista_wine.iloc[i:i+1,tam+2:tam+3]=hash_classif[3]
        lista_wine.iloc[i:i+1,tam+3:]=max(hash_classif, key=hash_classif.get)
        
    
    return lista_wine
#avaliando k=3
lista=selecionar_k_vizinhos_proximos(3,distance_matrix)
#classificar_vinho_knn_sem_peso(lista,test_train_set)
#Avaliando k=1
lista=selecionar_k_vizinhos_proximos(1,distance_matrix)
#classificar_vinho_knn_sem_peso(lista,test_train_set)
#eliminando útima coluna
new_wine_data=test_train_set.iloc[:,0:13]
#new_wine_data.head()
#matriz de distância
distance_matrix=montar_matriz_de_distancia(new_wine_data)
#distance_matrix.head()
#Avaliando k=1
lista=selecionar_k_vizinhos_proximos(1,distance_matrix)
#classificar_vinho_knn_sem_peso(lista,new_wine_data).head()
#Avaliando k=3
lista=selecionar_k_vizinhos_proximos(3,distance_matrix)
#classificar_vinho_knn_sem_peso(lista,new_wine_data)