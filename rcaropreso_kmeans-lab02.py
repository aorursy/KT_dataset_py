#CÉLULA KMEANS-LIB-01
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
#CÉLULA KMEANS-LIB-02
def tales(x1, x1_min, x1_max, x2_min, x2_max) :
    '''
    Esta função faz o escalamento de 1 amostra x1, que pertence ao intervalo [x1_min, x1_max], 
    dentro do intervalo [x2_min, x2_max]
    
    Parametros
    -----------
    x1     -> valor a ser normalizado
    x1_min -> valor minimo atingido por x1
    x1_max -> valor maximo atingido por x1
    x2_min -> valor minimo da escala de destino
    x2_max -> valor maximo da escala de destino    
    
    Retorno
    -----------  
    valor de x1, que pertence ao intervalo [x1_min, x1_max], projetado dentro do intervalo [x2_min, x2_max]
    '''
   
    #Teorema de Tales
    # ( x2 - x2_min ) / ( x2_max - x2_min ) = ( x1 - x1_min ) / (x1_max - % x1_min);
    
    #Definindo x1 como a escala do vetor dado e x2 como a escala de saída (variavel de interesse), isola-se x2:
    x2 = ( ( x1 - x1_min ) / (x1_max - x1_min) )* ( x2_max - x2_min ) + x2_min
    
    return x2
#CÉLULA KMEANS-LIB-03
def scale(x2_min, x2_max, sampleArray) :
    '''
    Esta função faz o escalamento de 1 vetor de amostras para dentro do intervalo [x2_min, x2_max]
    Ela retorna um vetor com os dados normalizados dentro do dominio dado
    
    Parametros
    -----------
    x2_min      -> valor minimo da escala de destino
    x2_max      -> valor maximo da escala de destino
    sampleArray -> vetor a ser normalizado   
    
    Retorno
    -----------  
    vetor sampleArray projetado dentro do intervalo [x2_min, x2_max]   
    '''    
    x_max = max(sampleArray)
    x_min = min(sampleArray)
    
    sampleArrayNorm = [tales(x, x_min, x_max, x2_min, x2_max) for x in sampleArray]
    
    return sampleArrayNorm
#CÉLULA KMEANS-LIB-04
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
#CÉLULA KMEANS-LIB-05
def calcBestK(N) :
    '''
    Calcula, a título de "chute inicial", um valor para o número de clusters (K)
    
    Parametros
    ----------
    N -> número de amostras disponiveis
    
    Retorno
    -------
    Número de clusters a serem usados
    '''
    return int(np.sqrt(N/2))
#CÉLULA KMEANS-LIB-06
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
#CÉLULA KMEANS-LIB-07
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
#CÉLULA KNN-LIB-01
from collections import OrderedDict
def kNN(vData, nClasses, value, k) :
    '''
    Define a classe a qual pertence um dado valor baseado na quantidade k de vizinhos mais próximos
    
    Parâmetros
    ----------
    vData    -> array contendo os dados que compõem as observações e suas respectivas denominações (labels)
    nClasses -> quantidade de classes (labels) utilizada
    value    -> dado a ser classificado através do k-NN
    k        -> quantidade de vizinhos necessários para a realização da votação (classificação)
    
    Retorno
    -------
    A função retorna um valor correspondente ao índice da classe vencedora ('0' -> classe 0, '1' -> classe 1, etc)
    '''
    
    distances = [ {'dist' : euclidean_distance(v1['point'], value), 'class': v1['class']} for v1 in vData ]
    print(distances)
    sortedItemsbyDistance = sorted(distances, key=lambda k: k['dist'])
    
    selectedNeighbours = sortedItemsbyDistance[0:k]
    
    vClasses = np.zeros(nClasses)
    
    for neighbour in selectedNeighbours :
        vClasses[ int(neighbour['class']) ] += 1
    
    print(selectedNeighbours)
    print(vClasses)
    
    return np.argmax(vClasses) #indice da classe vencedora ('0' -> classe 0, '1' -> classe 1, etc)
import seaborn as sns
#sns.pairplot(dfDataFile)



#IMPLEMENTE O CÓDIGO AQUI

#IMPLEMENTE O CÓDIGO AQUI

#IMPLEMENTE O CÓDIGO AQUI

#IMPLEMENTE O CÓDIGO AQUI
#Use 3 clusters

WSS =[] #Dispersão intra-cluster

BSS = [] #dispersão inter cluster
C = np.mean(dfArrayNorm, axis=0)

for nClusters in range(1,30) :
    #Dispersão intra-cluster
    (clusterData, vClusterCenters) = kmeans(dfArrayNorm, nClusters)
    cd = cluster_distortion(clusterData, nClusters)
    WSS.append(sum(cd))
    
    #Dispersão inter-cluster
    B=0
    for clusterId in range(len(vClusterCenters)) :
        vDataPoints = [ point for point in clusterData if point['clusterId'] == clusterId]
        nPoints = len(vDataPoints)
        clusterCenter = vClusterCenters[clusterId]
        
        B += nPoints * euclidean_distance(C, clusterCenter) ** 2
        
    BSS.append(B)
    
#print(WSS)
#IMPLEMENTE O CÓDIGO AQUI

#Verificando se os dados foram bem clusterizados
#Retorna a coluna de segmentos
dfTrainData['Customer_Segment'] = colCustomerSegment

#Fez meio fora de ordem (0, 2, 1)
#Ou seja, clusterId = 0 -> customer segment = 1
#Ou seja, clusterId = 2 -> customer segment = 2
#Ou seja, clusterId = 1 -> customer segment = 3
clusterCount = [0,0,0]
# c0_Count = 0
# c1_Count = 0
# c2_Count = 0
clusterDataError = []
for i, idVal in enumerate(vClusters):
    if( idVal == 0 and dfTrainData['Customer_Segment'][i] == 1) :
        clusterCount[idVal] += 1
    elif( idVal == 1 and dfTrainData['Customer_Segment'][i] == 3) :
        clusterCount[idVal] += 1
    elif( idVal == 2 and dfTrainData['Customer_Segment'][i] == 2) :
        clusterCount[idVal] += 1
    else:
        clusterDataError.append({'idCluster': idVal, 'customerSegment':dfTrainData['Customer_Segment'][i], 'data':dfTrainData.iloc[i, :]})
    
clusterCount
print(sum(dfTrainData['Customer_Segment'] == 1))
print(sum(dfTrainData['Customer_Segment'] == 2))
print(sum(dfTrainData['Customer_Segment'] == 3))