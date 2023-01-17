# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importando las librerias necesarias

from sklearn.datasets import fetch_mldata

import pandas as pd

import numpy

# Cargando los datos

mnist_t = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')

mnist_test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
# Ver los datos

mnist_t.head(10)

# Se obtiene que el label esta en la primera fila
# Procedemos a separar x y y

x_train = mnist_t.loc[:,'1x1':]

y_train = mnist_t.loc[:,'label']

x_test = mnist_test.loc[:,'1x1':]

y_test = mnist_test.loc[:,'label']
# Necesitamos estandarizar las dimensiones, a que pca toma en cuenta esto

# Para eso tenemos el standardscaler de sklearn

from sklearn.preprocessing import StandardScaler

#Inicializar

scaler = StandardScaler()

# Fitearlo en el x_train

scaler.fit(x_train)

# Transformar el x_train y el x_test

x_n_train = scaler.transform(x_train)

x_n_test = scaler.transform(x_test)
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

import time

var = [.99, .95, .92, .90]

comp = list()

seg = list()

acc = list()

for i in range(5):

    if i==0:

        pca = PCA()

    else:

        pca = PCA(var[i-1])

        

    # Fitearlo solo al training set

    pca.fit(x_n_train)

    

    # Transformar este

    x_nn_train = pca.transform(x_n_train)

    x_nn_test = pca.transform(x_n_test)

    

    # Anadimos componentes a nuestra lista de componentes

    comp.append(pca.n_components_)

    

    # Inicializamos logistic regression

    logisticRegr = LogisticRegression(solver='lbfgs')

    

    # Entrenamos con el training set  y calculamos el tiempo de entrenamiento

    t1 = time.time()

    logisticRegr.fit(x_nn_train,y_train)

    seg.append(round(time.time()-t1, 3))

    

    # Anadimos el puntaje de instancia a nuestra lista de puntajes

    acc.append(logisticRegr.score(x_nn_test,y_test))
# Imprimimos los resultados

for i in range(5):

    print (i+1,"iteracion: ", "numero de componentes: ", comp[i], "tiempo de entrenamiento: ", seg[i], "Score: ", acc[i])

    

# Se puede apreciar que al reducir las dimenciones a las mas influyentes con el PCA, en primer lugar se obtiene menor tiempo de entrenamiento

# A su vez se muestra que con una varianza del 95% en los componentes aumneta su score a 0.9217.
import matplotlib.pyplot as plt



pca1 = PCA()

pca2 = PCA(.95)



dn = pca1.fit_transform(x_train)

rd = pca2.fit_transform(x_train)
aprox = pca2.inverse_transform(rd)



plt.figure(figsize=(8,4));



plt.subplot(1, 2, 1);

plt.imshow(numpy.array(x_train.loc[1,'1x1':]).reshape(28,28),

              cmap = plt.cm.gray, interpolation='nearest',

              clim=(0, 255));

plt.xlabel('784 components', fontsize = 14)

plt.title('Original Image', fontsize = 20);



# 154 principal components

plt.subplot(1, 2, 2);

plt.imshow(aprox[1].reshape(28, 28),

              cmap = plt.cm.gray, interpolation='nearest',

              clim=(0, 255));

plt.xlabel('331 components', fontsize = 14)

plt.title('95% of Explained Variance', fontsize = 20);