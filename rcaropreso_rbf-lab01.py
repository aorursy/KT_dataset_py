#CÉLULA RBF-LIB-01

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
#CÉLULA RBF-LIB-02

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
#CÉLULA RBF-LIB-03

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

    return np.sqrt(N/2)
v1 = np.array([1,1])

v2 = np.array([4,5])

euclidean_distance(v1,v2)
#CÉLULA RBF-LIB-04

def plot_cluster(vInputs, vOutputs, W_1, sigma) :

    '''

    Plota o gráfico do cluster gerado pelo k-means e os raios de cada um.

    

    Parametros

    -----------

    vInputs   -> vetor/matriz com dados observados

    vOutputs  -> vetor/matriz com as saidas

    W_1       -> array com os centroides

    sigma     -> raio do cluster 

    

    '''

    c1X = []

    c1Y = []

    c2X = []

    c2Y = []

    for k in range(len(vInputs)) :

        if( vOutputs[k] == 1 ):

            c1X.append(np.asscalar(vInputs[k][0]))

            c1Y.append(np.asscalar(vInputs[k][1]))

        else:

            c2X.append(np.asscalar(vInputs[k][0]))

            c2Y.append(np.asscalar(vInputs[k][1]))



    fig = plt.figure(figsize=(10,10))

    plt.grid()

    plt.xlabel('X1')

    plt.ylabel('X2')

    plt.title('Clusters - RBF')

    plt.plot(c1X, c1Y, 'bx')

    plt.hold(True)

    plt.plot(c2X, c2Y, 'ro')

    

    #Plota os centroides

    plt.plot( W_1[:, 0], W_1[:, 1], 'k+')

    

    angle = np.arange(0, 2*np.pi, 0.01)

    for i in range( np.shape(W_1)[0]) :

        r = sigma[i]

        x = W_1[i][0]

        y = W_1[i][1]

        xp=r*np.cos(angle)

        yp=r*np.sin(angle)

        plt.plot(x+xp,y+yp, 'k')    
#CÉLULA RBF-LIB-05

def kmeans(vData, nClusters) :

    '''

    Este método calcula os centroides dos clusters de um conjunto de dados a partir do algoritmo k-means

    

    Parâmetros

    ----------

    vData -> vetor de coordenadas N-Dimensionais dos dados. Cada posição do array deve ser uma lista de coordenadas

        

    Retorno

    -------

    Array contendo a lista de pontos observados, onde cada posição do array corresponde a um clusterId. 

    Cada ponto do array de retorno é definido como um dicionário com a seguinte estrutura:

        {

         'pointCoord'  : coordenadas do ponto, 

         'centerCoord' : coordenadas de seu centroide, 

         'clusterId'   : identificador do cluster

        }

    '''



    #Inicializa os centroides

    vClusterCenters = [vData[i] for i in range(0, nClusters)]

    

    vPoints = [{'point':point, 'clusterId' : -1, 'centerCoord' : point*0} for point in vData] #tupla (point, clusterCentroid)

    

    clusterIsChanging = True

    while(clusterIsChanging == True) :

    

        clusterIsChanging = False



        vClusterPoints  = [[]] * nClusters #cria lista de listas com tamanho nClusters. O indice primario corresponde a cada cluster



        for dataPoint in vPoints :

            vDistances = [ euclidean_distance(dataPoint['point'], center) for center in vClusterCenters ]

            clusterIndex = np.argmin(vDistances)



            if( dataPoint['clusterId'] != clusterIndex ) :

                dataPoint['clusterId']  = clusterIndex

                clusterIsChanging = True



        #Após a redistribuição dos pontos, recalcula os centroides

        for clusterIndex, clusterCenter in enumerate(vClusterCenters) :

            clusterData = [ point['point'] for point in vPoints if point['clusterId'] == clusterIndex ]

            vClusterCenters[clusterIndex] = np.mean(clusterData, axis=0) #atualiza centroide do cluster indicado pelo clusterIndex



    

    for dataPoint in vPoints :

        dataPoint['centerCoord'] = vClusterCenters[dataPoint['clusterId']]

                    

    return (vPoints, vClusterCenters)
#CÉLULA RBF-LIB-06

def cluster_distortion(vDataPoints, nClusters, bWeighted = False) :

    '''

    Esta função calcula a distorção intra-cluster

    

    Parâmetros

    ----------

    vDataPoints -> array contendo os pontos (dados observados). Cada ponto é definido como um dicionário com a seguinte estrutura

        {'pointCoord' : coordenadas do ponto, 'centerCoord' : coordenadas de seu centroide, 'clusterId' : identificador do cluster}

    

    nClusters   -> quantidade de clusters utilizados

    bWeighted   -> Se for definido como True, cada distorção será ponderada pelo numero de dados do cluster, resultando no 

                   cálculo da distância média quadrática (ou variância da função de ativação da RBF).

    

    Retorno

    -------

    

    Um array contendo a distorção intra-cluster de todos os clusters, onde cada posição do array corresponde ao clusterId    

    '''

    

    vClusterDist = []

    for i in range(nClusters) :

        vClusterPoints = [point for point in vDataPoints if point['clusterId'] == i]

        distance = sum([euclidean_distance(point['point'], point['centerCoord'])**2 for point in vClusterPoints])



        if( bWeighted == True) :

            distance /= len(vClusterPoints)

        

        vClusterDist.append(distance)

    

    return vClusterDist    
#CÉLULA RBF-LIB-07

def loadDataFile(strFileName) :

    '''

    Esta função carrega os dados de um arquivo e retorna um dataframe

    '''

    return pd.read_csv(strFileName)    
#CÉLULA RBF-LIB-08

def avgSquaredError(x, d, W_2) :

    '''

    Calula o Erro Quadratico Medio (ASE)

    

    Parametros

    ----------

    x   -> array de dados da entrada

    d   -> array de dados de saída

    W_2 -> pesos conectando as camadas de entrada e saida

    

    Retorno

    -------

    Valor do EQM para uma época de treinamento.    

    '''

    nSamples = np.shape(x)[1]

    

    I_2 = np.matmul(W_2, x)    

    Y_2 = I_2 #Função de ativação linear

    

    #IMPLEMENTE O CÓDIGO ABAIXO

    #E_k =

    #E_k =        #soma os elementos de cada coluna entre si

    #ASE =        # totaliza os elementos do array fazendo a media

    

    return 0
v1 = np.array([[-1.00000000e+00,  7.65315335e-01,  5.31685471e-04,

         1.35633610e-06,  1.49699587e-09,  8.18411773e-02],

       [-1.00000000e+00,  2.10729788e-06,  1.85049740e-03,

         8.98790304e-01,  7.59415283e-03,  2.02347730e-03],

       [-1.00000000e+00,  5.37376118e-02,  2.27668290e-04,

         9.93741764e-04,  3.21429752e-03,  9.86359559e-01],

       [-1.00000000e+00,  3.31508135e-04,  7.44654735e-06,

         1.14956399e-02,  1.79980697e-01,  3.63889345e-01],

       [-1.00000000e+00,  5.10030797e-01,  7.56032639e-05,

         7.14297576e-07,  2.33230128e-09,  1.21011447e-01],

       [-1.00000000e+00,  7.26809402e-01,  6.38797877e-02,

         7.56738299e-05,  6.73357809e-08,  6.12866773e-02],

       [-1.00000000e+00,  1.82820236e-03,  2.51183550e-02,

         3.89269583e-01,  2.63098662e-01,  1.28857162e-01],

       [-1.00000000e+00,  3.02852033e-04,  5.39220246e-03,

         5.58905042e-01,  4.65386245e-01,  7.90828075e-02],

       [-1.00000000e+00,  4.84793191e-02,  1.02258260e-02,

         2.36300777e-02,  5.67325198e-02,  7.26608497e-01],

       [-1.00000000e+00,  4.34817699e-02,  5.76015979e-06,

         2.34828888e-05,  2.23944072e-05,  6.24902695e-01]])



v2 = np.array([-1,  1, -1,  1, -1, -1,  1,  1, -1, -1])



WW = np.array([[ 0.12686105, -1.19113089, -0.08158087,  1.46640788,  1.33961452,

        -0.80448037]])



print(avgSquaredError(v1.T, v2, WW))
#Resultado esperado: 0.09050036059038595