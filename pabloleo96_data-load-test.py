# Pablo Leo Muñoz
# Imports necesarios

import numpy as np 

import os

import cv2

import matplotlib.pyplot as plt
# Directorios del conjunto de entrenamiento y test

train_dir = "../input/train/train/"

test_dir = "../input/test_mixed/Test_Mixed/"
# Obtenemos las clases disponibles (nombres de las carpetas)

clases = os.listdir(train_dir)

print("Existen un total de {} clases de corales".format(len(clases)))

print("Las clases son: {}".format(clases))
# Cargamos el conjunto de datos de entrenamiento en memoria (hay ~17GB en la máquina, nos sobra)

x_train = np.array([cv2.imread(os.path.join(train_dir, cl, name)) for cl in clases

           for name in os.listdir(os.path.join(train_dir, cl))])

y_train = np.array([cl for cl in clases

           for name in os.listdir(os.path.join(train_dir, cl))])
# obtenemos un índice aleatorio

idx = np.random.randint(len(x_train))



plt.imshow(x_train[idx]) # mostramos la imágen

plt.title(y_train[idx]) # indicamos la clase del coral cargado en el título

plt.show()