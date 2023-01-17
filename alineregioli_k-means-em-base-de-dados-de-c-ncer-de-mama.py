# Carregar as bibliotecas necessárias: 



import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score 



from sklearn import metrics



from sklearn.cluster import KMeans





# Carregar a base de dados:



rout_path = "../input/data.csv"

dados = pd.read_csv(rout_path)
# Mostrar detalhes dos 5 primeiros registros da base:



dados.head() 
# Colocar no vetor Y os valores referentes às classes



Y = dados.diagnosis  





# Fazer a remoção das colunas desnecessárias



list = ['Unnamed: 32','id','diagnosis']        # lista com as colunas a serem removidas

X = dados.drop(list,axis = 1 )          

X.head()

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



#Para aplicar PCA, primeiro é preciso fazer a transformação dos dados

x = StandardScaler().fit_transform(X)

y = Y



#Aplicar PCA

pca = PCA(n_components=2) #escolhe a quantidade de componentes

principalComponents = pca.fit_transform(x) #aplica nos dados
from sklearn.decomposition import PCA

pca = PCA(n_components=2) #escolhe a quantidade de componentes

principalComponents = pca.fit_transform(x) #aplica nos dados

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, dados.diagnosis], axis = 1)



#plotar os dados reais:

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

targets = ['B', 'M']

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['diagnosis'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
#Confrontar K vs Inertia

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n, random_state= 111  , algorithm='auto'))

    algorithm.fit(principalComponents)

    inertia.append(algorithm.inertia_) # inertia_ eh a distancia euclidiana ao centroide mais proximo, logo, sempre sera descrescente

    

    

plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Número de clusters (K)') , plt.ylabel('Inertia')

plt.show()
#Aplicar K-Means com K=2



algorithm = (KMeans(n_clusters = 2, random_state= 111 ) )

algorithm.fit(principalComponents)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_

#Mostrar como ficou depois de clusterizado com k=2:

h = 0.02

x_min, x_max = principalComponents[:, 0].min() - 1, principalComponents[:, 0].max() + 1

y_min, y_max = principalComponents[:, 1].min() - 1, principalComponents[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])



plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'principal component 1' ,y = 'principal component 2' , data =principalDf, c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('PC1') , plt.xlabel('PC2')

plt.show()
labels_pred = np.where(labels1==1, 'B', 'M') #transformar 1 e 0 em B e M



#transformar em listas:

predito = labels_pred.tolist()

esperado = Y.tolist()



#Calcular porcentagem

certos = 0

total_exemplos = len(predito)

for i in range(0, total_exemplos):

  if ( predito[i] == esperado[i]):

    certos = certos + 1

print(certos, "/" , total_exemplos)

acerto =  certos/len(predito) 





print("Porcentagem de acerto = ", acerto)

#Silhouette

print("Silhouette =", metrics.silhouette_score(principalComponents, labels_pred, metric='euclidean') ) #Deu bom pq os clusters sao densos

from sklearn.metrics import davies_bouldin_score

#Davies Bouldin

print("Davies Bouldin = ", davies_bouldin_score(principalComponents, labels_pred) )  #não deu bom pq os clusters tao bem juntinhos