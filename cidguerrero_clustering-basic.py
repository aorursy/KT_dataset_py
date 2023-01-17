import numpy as np

from scipy.io import loadmat



def findClosestCentroids(X, centroids):



    # Calculamos el número de centroides K

    K = centroids.shape[0]



    # Calculamos el número de ejemplos m

    m = X.shape[0]



    # Tienes que devolver el valor correcto de idx. Lo inicializamos en cero.

    idx = np.zeros(m, dtype=int)



    # ====================== COMPLETA TU CÓDIGO ABAJO ======================

    # Instrucciones: Recorre cada ejemplo, encuentra su centroide más cercano,

    # y almacena el índice en la ubicación correcta de idx. Específicamente,

    # idx es un vector de dimensiones m x 1, donde idx(i) debe contener el índice

    # del centroide más cercano al ejemplo i. Por lo tanto, debe contener valores

    # en el rango 0...(K-1)

    #

    # Nota: Puedes usar bucles 'for' 



    for i in range(m):

        

        distancia_mas_cercana = None



        for j in range(K):

            x_menos_mu = X[i] - centroids[j]

            distancia = np.sum(x_menos_mu * x_menos_mu)

            

            if ((distancia_mas_cercana is None) or (distancia < distancia_mas_cercana)):

                idx[i] = j

                distancia_mas_cercana = distancia

                

    # =============================================================



    return idx
print('Encontrando los centroides más cercanos... \n');



# Cargar el conjunto de datos que usaremos

#os.listdir("../input/dataset")

exdata2 = loadmat('../input/ex7data2.mat')

X = np.array(exdata2['X'])



# Seleccionar un conjunto inicial de centroides

K = 3  # 3 centroides

initial_centroids =  np.array([(3., 3.), (6., 2.), (8., 5.)])



# Encontrar los centroides iniciales más cercanos a los ejemplos

idx = findClosestCentroids(X, initial_centroids)



print('Centroides más cercanos a los primeros 3 ejemplos: ', idx[0:3])

print('(los centroides más cercanos deberían ser 0, 2 y 1, respectivamente)\n');
print('Centroides más cercanos a los últimos 3 ejemplos: ', idx[-3:])
def computeCentroids(X, idx, K):



    # Calculamos el número de ejemplos m y el número de características n

    (m, n) = X.shape



    # Tienes que devolver el valor correcto de cada centroide. Los inicializamos en cero.

    centroids = np.zeros((K, n))



    # ====================== COMPLETA TU CÓDIGO ABAJO ======================

    # Instrucciones: Recorre cada centroide y calcula la media de todos los puntos

    # que le han sido asignados. Especificamente, el vector fila centroids[i]

    # debe contener la media de los puntos de datos asignados al centroide i.

    #

    # Nota 1: Puedes usar bucles 'for' 

    # Nota 2: X[idx == 2] es el subconjunto de puntos de datos asignados al centroide 2.

    # Nota 3: Se puede calcular la media con la función numpy.mean(). No olvidar asignar

    #         el parámetro axis=0 para calcular la media de las filas.

    # 



    for i in range(K):

        centroids[i] = X[idx == i].mean(axis=0)



    # =============================================================



    return centroids
print('Recalculando la posición de los centroides... \n');



# Recalcular los centroides a partir de las asignaciones realizadas en el paso anterior

centroids = computeCentroids(X, idx, K)



print('Centroides calculados después de encontrar reasignar los puntos al centroide más cercano:')

print(centroids)



print('\n(Los dos primeros centroides deberían ser:')

print('[[ 2.42830111  3.15792418]')

print(' [ 5.81350331  2.63365645]] )')
X[23:25]
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



def runkMeans(X, initial_centroids, max_iters, plot_progress=False):

    

    # Inicializar los valores

    (m, n) = X.shape

    K = initial_centroids.shape[0]

    centroids = initial_centroids

    idx = np.zeros(m, dtype=int)



    # Llevaremos un historial de los centroides por cada cluster y cada iteración

    centroids_history = np.zeros((K, max_iters + 1, n))

    centroids_history[:,0] = initial_centroids



    # Ejecutar k-means

    for i in range(max_iters):

    

        # Estatus de avance

        print('Iteración de k-means %d/%d...' % ((i+1), max_iters))

        

        # Asignar cada de ejemplo en X a su centroide más cercano

        idx = findClosestCentroids(X, centroids)

        

        # Opcionalmente, mostrar diagrama con avance

        if plot_progress:

            centroids_history[:,i] = centroids

            plotProgresskMeans(X, centroids, centroids_history, idx, K, i)

            input('Presiona Intro para continuar.')



        # Calcular los nuevos centroides luego de las asignaciones a clusters

        centroids = computeCentroids(X, idx, K)

        

    return (centroids, idx)
def plotProgresskMeans(X, centroids, centroids_history, idx, K, i):

    

    plt.title('Iteración #%d' % (i+1) )

    

    # Dibujar los ejemplos del conjunto de datos

    plt.scatter(X[:,0], X[:,1], c=idx, cmap='prism')

    

    # Dibujar el avance desde los centroides iniciales

    for j in range(K):

        plt.plot(centroids_history[j, 0:(i+1), 0], centroids_history[j, 0:(i+1), 1], '-', c='gray')

        plt.scatter(centroids_history[j, 0:i, 0], centroids_history[j, 0:i, 1], marker='d', c='gray')

    

    # Dibujar los centroides

    plt.scatter(centroids[:,0], centroids[:,1], c='k', marker='x')



    plt.show()



    return
print('Ejecutando clustering de k-means en el conjunto de datos... \n');



# Cargar el conjunto de datos

exdata2 = loadmat('../input/ex7data2.mat')

X = np.array(exdata2['X'])



# Parámetros de k-means

K = 3

max_iters = 10



# Por consistencia, configuramos acá unos valores específico pero en la práctica

# habrá que generarlos automáticamente de manera aleatoria, como se verá más

# adelante en kMeansInitCentroids

initial_centroids =  np.array([(3., 3.), (6., 2.), (8., 5.)])



# Ejecutar el algoritmo k-means

(centroids, idx) = runkMeans(X, initial_centroids, max_iters, plot_progress=True)

print('\nFin de ejecución de k-means.')
from numpy.random import permutation



def kMeansInitCentroids(X, K):

    

    # Calcular el número de ejemplos

    m = X.shape[0]

    

    # Reordernar aleatoriamente los índices de los ejemplos

    randidx = permutation(m)

    

    # Tomar los primeros K ejemplos como centroides

    centroids = X[randidx[0:K]]

    

    return centroids
print('Ejecutando clustering de k-means en los pixels de una imagen... \n');



# Cargar la imagen de un guacamayo

from scipy import misc



# Cargar la imagen y convertirla en un arreglo numpy

A = misc.imread('../input/macaw_small.jpg')

A = np.asarray(A, dtype=float)



# Dividir entre 255 para que los valores estén en el rango 0 - 1

A = A / 255



# Tamaño de la imagen

img_size = A.shape



# Convertir la imagen en una matrix N x 3, donde N = número de pixels.

# Cada fila contiene los valores de Rojo, Verde y Azul

# Esto nos da la matriz del conjunto de datos X que usaremos con k-means

X = A.reshape((img_size[0] * img_size[1], 3))



# Ejecuta tu algoritmo k-means en estos datos

# Prueba con diferentes valores de K y max_iters

K = 16

max_iters = 10



# Cuando se usa k-means, es importante inicializar aleatoriamente los centroides

# Usaremos la función kMeansInitCentroids

initial_centroids = kMeansInitCentroids(X, K)



# Ejecutar k-means

(centroids, idx) = runkMeans(X, initial_centroids, max_iters)
print('Aplicando k-means para comprimir una imagen... \n');



# Encontrar los centroides más cercanos a cada pixel

idx = findClosestCentroids(X, centroids)



# Esencialmente, hemos representado la imagen X en función 

# de los índices en idx



# Ahora podemos recuperar la imagen a partir de los índices (idx)

# mapeando cada pixel (identificado con su índice en idx) al valor

# de su centroide

X_recovered = centroids[idx]



# Transformar la imagen recuperada en sus dimensiones originales

X_recovered = X_recovered.reshape((img_size[0], img_size[1], 3))



# Mostrar la imagen original y la recuperada

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.axis('off')

ax1.imshow(A)

ax1.set_title('Original')

ax2.axis('off')

ax2.imshow(X_recovered)

ax2.set_title('Comprimida, con %d colores' % K)
