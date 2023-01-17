from sklearn.cluster import KMeans

import pandas as pd

import numpy as np
df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
df.head()
train = df.species
classes = df.drop('species', axis=1)
# Função que retorna a distância eucludiana de dois vetores de duas dimensões.

from sklearn.neighbors import DistanceMetric

def calcula_distancia(x,c):

    dist = DistanceMetric.get_metric('euclidean')

    return dist.pairwise(x,c)
v1 = [[1.2,1,2.1,1]]

v2 = [[1,1.9,5.4,3.2]]



calcula_distancia(v1,v2)
v3 = [[0.5,0,2.1,1.5]]

v4 = [[0.5,0,2.1,1.5]]



calcula_distancia(v3,v4)
kmeans = KMeans(n_clusters=3)
kmeans.fit(classes)

centros = kmeans.cluster_centers_

centros
import random
classes[34:35]
for i in range(1,4):

    num = round(random.uniform(1,50),0)

    num_int = int(num)

    print("Para a amostra de dado número número ", num_int," a distância para os centros é de: ", calcula_distancia(classes[(num_int-1):num_int],centros))

    print("\n")
distancia = kmeans.fit_transform(classes)

distancia
import seaborn as sns
sns.kdeplot(classes.sepal_length,shade=True )
sns.kdeplot(classes.sepal_width,shade=True )
sns.kdeplot(classes.petal_length,shade=True )
sns.kdeplot(classes.petal_width,shade=True )
X = []

num1 = random.uniform(4,8)

X.append(num1)

num2 = random.uniform(1.5,4.5)

X.append(num2)

num3 = random.uniform(0,8)

X.append(num3)

num4 = random.uniform(0,3)

X.append(num4)

X = np.array(X).reshape(1,4)

X
calcula_distancia(X,centros)
print(kmeans.predict(X))
i = 0

while i < 10:

    X = []

    num1 = random.uniform(4,8)

    X.append(num1)

    num2 = random.uniform(1.5,4.5)

    X.append(num2)

    num3 = random.uniform(0,8)

    X.append(num3)

    num4 = random.uniform(0,3)

    X.append(num4)

    X = np.array(X).reshape(1,4)

    print("Dado", i+1, "gerado: ", X,"Distancia para os centros de: ", calcula_distancia(X,centros)," Classificação prevista: ", kmeans.predict(X))

    i = i + 1