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
data_wine= pd.read_csv("/kaggle/input/wine-daset/wineDATA.CSV")

#Motando conjunto de teste e treino
def set_train_test(frac_train,data_wine):
    
    #selecionando aleatoriamente 50% da base para treino
    #replace=False não terá repetição de dados
    train_set= data_wine.sample(frac=frac_train, replace=False)
    
    
    #retirando sobreposição para o conjunto de teste
    combined = data_wine.append(train_set)
    test_set= combined[~combined.index.duplicated(keep=False)]
    
    
    return train_set,test_set

train_set,test_set=set_train_test(0.5,data_wine)
#conjunto de treinamento e teste
#train_set,test_set


#motando a mastriz de distância euclidiana 
def montar_matriz_de_distancia(test_set,train_set):
    distance_matrix= spatial.distance.cdist(test_set.iloc[:,1:],
                                            train_set.iloc[:,1:], metric='euclidean')
    
    #coluna representa o indice do conjunto de treino e linha o indece do conjunto de teste
    return pd.DataFrame(distance_matrix,columns=train_set.index,index=test_set.index)

    
#selecionando k_nn vizinhos mais próximo
def selecionar_k_vizinhos_proximos(quant_vizinho,distance_matrix):
    trans_matrix=distance_matrix.T
    lista=[]
    for k in distance_matrix.index:
        #selecionando o vizinho mais próximo
        ex_kNN=trans_matrix[k].nsmallest(n=quant_vizinho, keep='first')
        lista.append(ex_kNN)
    return  lista

#classificar espécie do conjunto de teste utilizando k_nn sem peso  
def classificar_class_knn_com_peso(lista_k_nn,train_set,test_set):
    
    lista_classif=[]
    lista_index_class=pd.DataFrame(lista_k_nn).index
    for i in range(len(lista_k_nn)):
        
        #recuperando index da Class classificada com menor peso
        lista=lista_k_nn[i].index 
        
        #inicializando dicionário para guardar os pesos a cada iteração
        dict_classif={'1':0,'2':0,'3':0}
        frame={'Class':'','Classificacao':''}
        
        for k in range(len(lista)):
            
            #recuperando a classe do menor peso atual
            classif=train_set.loc[lista[k]]['Class']
            
            #recuperando o índice do menor peso atual
            indice=lista[k]
            
            #clacular os pesos
            if (lista[k]!=0):
                if(classif==1):
                    dict_classif['1']+=(1/lista_k_nn[i][indice])

                elif(classif==2):
                     dict_classif['2']+=(1/lista_k_nn[i][indice])

                elif(classif==3):
                     dict_classif['3']+=(1/lista_k_nn[i][indice])
                        
        #classificação               
        frame['Classificacao']=max(dict_classif, key=dict_classif.get)
        #recuperando a classe correta
        frame['Class']=int(test_set.loc[lista_index_class[i]]['Class'])
       
       
        
        lista_classif.append(frame)
                
        
    return pd.DataFrame(lista_classif)


#Calculando a taxa de acerto do classificador
def calcular_taxa_acerto_classificador(classif):
    ac_1=len(classif[(classif['Classificacao']=='1') &
                             (classif['Class']==1)])
    ac_2=len(classif[(classif['Classificacao']=='2') &
                             (classif['Class']==2)])
    ac_3=len(classif[(classif['Classificacao']=='3') &
                             (classif['Class']==3)])
    
    return (ac_1+ac_2+ac_3)/len(classif)*100

def holdout_aleatorio_com_repeticao(n):
    
    dict_taxa={}
    lista_acerto=[]
    for k in range(n):
        
        #inicializando dicionário para guardar a taxa de acerto
        dict_taxa={'taxa_acerto_classificador1':0,'taxa_acerto_classificador3': 0, 'diferença_taxa_acerto':0}
        # conjunto de treino e teste
        train_set,test_set=set_train_test(0.5,data_wine)
        #matriz de distância
        distance_matrix=montar_matriz_de_distancia(test_set,train_set)
        
        #1_nn 
        lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
        #classificando 
        classif=classificar_class_knn_com_peso(lista_k_nn,train_set,test_set)
        #calculando taxa de acerto do classificador
        dict_taxa['taxa_acerto_classificador1']=calcular_taxa_acerto_classificador(classif)
        
        #3_nn
        lista_k_nn=selecionar_k_vizinhos_proximos(3,distance_matrix)
        #classificando 
        classif=classificar_class_knn_com_peso(lista_k_nn,train_set,test_set)
        #calculando taxa de acerto do classificador
        dict_taxa['taxa_acerto_classificador3']=calcular_taxa_acerto_classificador(classif)
        
        #calculando a diferença da taxa de acerto 
        dict_taxa['diferença_taxa_acerto']=dict_taxa['taxa_acerto_classificador1']-dict_taxa['taxa_acerto_classificador3']
        
        lista_acerto.append(dict_taxa)
    return pd.DataFrame(lista_acerto)
classif_holdout=holdout_aleatorio_com_repeticao(100)

classif_holdout


#calculando a média de acerto
media_diferença_acerto=classif_holdout['diferença_taxa_acerto'].mean()
#calculando o desvio padrão
desvio_padrao=statistics.stdev(classif_holdout['diferença_taxa_acerto'])
#intervalo de confiança
["%.2f" %(media_diferença_acerto-1.96*desvio_padrao),"%.2f" %(media_diferença_acerto+1.96*desvio_padrao)]

#calculando a média de acerto do classificado 1_nn com peso
media_acerto1=classif_holdout['taxa_acerto_classificador1'].mean()
#calculando o desvio padrão
desvio_padrao1=statistics.stdev(classif_holdout['taxa_acerto_classificador1'])
#intervalo de confiança
inter1,intert1_2=["%.2f" %(media_acerto1-1.96*desvio_padrao1),"%.2f" %(media_acerto1+1.96*desvio_padrao1)]

#calculando a média de acerto do classificado 3_nn com peso
media_acerto3=classif_holdout['taxa_acerto_classificador3'].mean()
#calculando o desvio padrão
desvio_padrao3=statistics.stdev(classif_holdout['taxa_acerto_classificador3'])
#intervalo de confiança
inter3,intert3_2=["%.2f" %(media_acerto3-1.96*desvio_padrao3),"%.2f" %(media_acerto3+1.96*desvio_padrao3)]


#visualização
resultado = { 'Média da taxa de acerto 1_nn':"%.2f" %  media_acerto1,
            'Intervalo de confiança 1_nn':   str(inter1)+"__"+str(intert1_2),
             'Média da taxa de acerto 3_nn':"%.2f" %  media_acerto3,
            'Intervalo de confiança 3_nn':  str(inter3)+"__"+str(intert3_2)}
             

resultado

