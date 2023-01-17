#CÉLULA SCIKIT-LIB-01

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
#CÉLULA KMEANS-LIB-02

def euclidean_distance(v1, v2) :

    '''

    Esta função recebe 2 arrays (do tipo np.array) e retorna a distância euclidiana entre eles

    

    Parâmetros

    ----------

    v1 -> vetor de coordenadas do primeiro ponto

    v2 -> vetor de coordenadas do segundo ponto

    

    Retorno

    -------

    Distância entre os dois pontos    

    '''

    return np.sqrt( sum((v1 - v2)**2) )
#CÉLULA KMEANS-LIB-03

def kmeans(vData, nClusters) :

    '''

    Este método calcula os centroides dos clusters de um conjunto de dados a partir do algoritmo k-means

    

    Parâmetros

    ----------

    vData -> vetor de coordenadas N-Dimensionais dos dados. Cada posição do array deve ser uma lista de coordenadas

        

    Retorno

    -------

    Uma tupla (vPoints, vClusterCenters), onde:

    

    vPoints -> é um Array contendo a lista de pontos observados, onde cada posição do array corresponde a um clusterId. 

    Cada ponto do array de retorno é definido como um dicionário com a seguinte estrutura:

        {

         'pointCoord'  : coordenadas do ponto, 

         'centerCoord' : coordenadas de seu centroide, 

         'clusterId'   : identificador do cluster

        }

        

    vClusterCenters -> é uma lista contendo as coordenadas de cada centróide, ordenadas de acordo com o índice 'clusterId'

    '''



    #Inicialização sequencial dos centroides

    #vClusterCenters = [vData[i] for i in range(0, nClusters)]    

    

    #Inicialização usando kmeans++

    vClusterCenters = [vData[0]]

    for k in range(1, nClusters):

        D2 = np.array([min([np.inner(c-x,c-x) for c in vClusterCenters]) for x in vData])

        probs = D2/D2.sum()

        cumprobs = probs.cumsum()

        r = np.random.rand()

        for j,p in enumerate(cumprobs):

            if r < p:

                i = j

                break

        vClusterCenters.append(vData[i])

        

    vPoints = [{'point':point, 'clusterId' : -1, 'centerCoord' : point*0} for point in vData]

    

    clusterIsChanging = True

    while(clusterIsChanging == True) :

    

        clusterIsChanging = False



        for dataPoint in vPoints :

            #Calcula a distancia entre o ponto 'dataPoint' e todos os centroides

            vDistances = [ euclidean_distance(dataPoint['point'], center) for center in vClusterCenters ]

            #Define a menor distância

            clusterIndex = np.argmin(vDistances)



            #Verifica se houve mudança de atribuição e ativa o flag em caso afirmativo            

            if( dataPoint['clusterId'] != clusterIndex ) :

                dataPoint['clusterId']  = clusterIndex

                clusterIsChanging = True



        #Após a redistribuição dos pontos, recalcula os centroides

        for clusterIndex, clusterCenter in enumerate(vClusterCenters) :

            #Realiza o agrupamento dos pontos de um dado cluster

            clusterData = [ point['point'] for point in vPoints if point['clusterId'] == clusterIndex ]

            #atualiza centroide do cluster indicado pelo clusterIndex

            vClusterCenters[clusterIndex] = np.mean(clusterData, axis=0) 



    #Atualiza as coordenadas dos centroides calculados pelo algoritmo   

    for dataPoint in vPoints :

        dataPoint['centerCoord'] = vClusterCenters[dataPoint['clusterId']]



    return (vPoints, vClusterCenters)
#CÉLULA KMEANS-LIB-04

def cluster_distortion(vDataPoints, nClusters) :

    '''

    Esta função calcula a distorção intra-cluster

    

    Parâmetros

    ----------

    vDataPoints -> array contendo os pontos (dados observados). Cada ponto é definido como um dicionário com a seguinte estrutura

        {'pointCoord' : coordenadas do ponto, 'centerCoord' : coordenadas de seu centroide, 'clusterId' : identificador do cluster}

    

    nClusters   -> quantidade de clusters utilizados

    

    

    Retorno

    -------

    

    Um array contendo a distorção intra-cluster de todos os clusters, onde cada posição do array corresponde ao clusterId    

    '''

    vClusterDist = []

    

    for i in range(nClusters):

        #Realiza o agrupamento dos pontos de um dado cluster

        vClusterPoints = [point for point in vDataPoints if point['clusterId']==i]

        

        #Calcula a soma das distâncias (elevadas ao quadrado) entre um cada ponto e o centro de seu respectivo cluster. 

        distance = sum([euclidean_distance(point['point'], point['centerCoord'])**2 for point in vClusterPoints])

        vClusterDist.append(distance)

        

    return vClusterDist