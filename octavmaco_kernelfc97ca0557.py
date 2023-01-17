# Importul bibliotecilor pentru generarea setului de date și vizualizarea acestuia

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs



# Crearea setului de date prin generarea de valori aleatoare

X, y = make_blobs(

   n_samples=150, n_features=2,

   centers=3, cluster_std=0.5,

   shuffle=True, random_state=0

)



# Reprezentarea grafică a punctelor din setul de date

plt.scatter(

   X[:, 0], X[:, 1],

   c='white', marker='o',

   edgecolor='black', s=50

)

plt.show()
# importul metodei KMeans din obiectul cluster al bibliotecii scikit-learn

from sklearn.cluster import KMeans



# Inițializarea algoritmului în variabila km

km = KMeans(

    n_clusters=3, init='random',

    n_init=10, max_iter=300, 

    tol=1e-04, random_state=0

)

y_km = km.fit_predict(X)
# Reperzentarea grafică a celor 3 clustere utilizând biblioteca matplotlib

plt.scatter(

    X[y_km == 0, 0], X[y_km == 0, 1],

    s=50, c='lightgreen',

    marker='s', edgecolor='black',

    label='Cluster 1'

)



plt.scatter(

    X[y_km == 1, 0], X[y_km == 1, 1],

    s=50, c='orange',

    marker='o', edgecolor='black',

    label='Cluster 2'

)



plt.scatter(

    X[y_km == 2, 0], X[y_km == 2, 1],

    s=50, c='lightblue',

    marker='v', edgecolor='black',

    label='Cluster 3'

)



# Reprezentarea grafică a centroizilor aferenți

plt.scatter(

    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],

    s=250, marker='*',

    c='red', edgecolor='black',

    label='Centroizi'

)

plt.legend(scatterpoints=1)

plt.grid()

plt.show()
# Calcularea distorsiunii pentru o serie de valori k privind numărul clusterelor

distortions = []

for i in range(1, 11):

    km = KMeans(

        n_clusters=i, init='random',

        n_init=10, max_iter=300,

        tol=1e-04, random_state=0

    )

    km.fit(X)

    distortions.append(km.inertia_)



# Reprezentarea grafică a distorsiunii în funcție de numărul k de clustere

plt.plot(range(1, 11), distortions, marker='o')

plt.xlabel('Numarul de clustere sau grupuri')

plt.ylabel('Distorsiunea')

plt.show()