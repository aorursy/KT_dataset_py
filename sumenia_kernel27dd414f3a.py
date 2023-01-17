X_train = []
Y_train = []
file_name = []
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from keras.preprocessing.image import load_img

import os # package pour interagir avec le système oú est executer ce code
# ca permet par exemple de parcourir des dossiers pour acceder a des fichiers

number = 0
for dirname, _, filenames in os.walk('/kaggle/input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        file_name.append(filename[:-5])
        img = load_img(file_path, target_size=(1024, 1024), interpolation='bilinear')
        X_train.append(img)
        number = number + 1
        if (number >= 200) :
            break


from matplotlib import pyplot


pyplot.imshow(X_train[4])
import csv

csvfile = open('/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv', 'r', newline='') # on charge le fichier en mêmoire
labels_csv = csv.reader(csvfile) # on recupere dans une variable la structure de donnée CSV

labels_dict = {} # je creeer un dictionnaire vide : KEY -> VALEUR

for row in labels_csv : # je parcours mon CSV
   labels_dict[row[0]] = row[1] # KEY = la premiere colone du CSV ; VALEUR = la deuxieme colone du CSV

Y_train = [] # je creer une LIST vide

for file in file_name : # je parcours DANS L'ORDRE les images qu'on a chargé en mémoire
        label = labels_dict[file] # je cherche le label asocié a l'image dans mon dictionnaire
        Y_train.append(label) # je rajoute dans la liste le LABEL dans l'ORDRE

csvfile.close() # je ferme le fichier CSV qui a ete ouvert car on en a plus besoin
# X_train --> TOUTES LES PHOTOS SOUS UN FORMAT DE DONNÉES "PIL", DU PACKAGE "PILLOW", INTEGRER A KERAS
# PIL permet de changer la resolution, de changer de PNG a JPEG par exemple ou de passer de la couleur au noir et blanc
# Y_train --> TOUT LES LABELS DANS LE MEME ORDRE QUE LES PHOTOS
import keras
# JE VEUX TRANSFORMER MA LIST D'IMAGE AU FORMAT PIL EN UN NUMPY ARRAY
from keras.preprocessing.image import img_to_array
import numpy as np

# NE FONCTIONNE PAS, IL Y A UN BUG DANS KERAS NON CORRIGER
# ON AURAIT DU UTILISER TF.KERAS ET PAS KERAS BRUT
#X_train = img_to_array(X_train) # JE m'attendais a recevoir une list de numpy array

list_of_numpy_array = [] # on veut une list où chaque element contient l'image au format NUMPY
for image in X_train :
    img_as_numpy = img_to_array(image) # PIL -> NUMPY
    list_of_numpy_array.append(img_as_numpy)

# print(len(list_of_numpy_array))
# print(list_of_numpy_array[0].shape)

# plutot que d'avoir 200 elements au format (1024, 1024, 3), on veut une matrice au format (200, 1024, 1024, 3)

training_images_as_numpy = np.array(list_of_numpy_array)
print(training_images_as_numpy.shape) # (200, 1024, 1024, 3)

# on a diviser les images par 255.0 pour passer d'un interval [0, 255] intervale à un interval [0, 1]
training_images_as_numpy = training_images_as_numpy / 255.0

label_as_numpy = np.array(Y_train).reshape(200, 1)
print(label_as_numpy.shape)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(1024, 1024, 3))) # input layer
model.add(keras.layers.MaxPool2D((2, 2)))

model.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(1024, 1024, 3))) # input layer
model.add(keras.layers.MaxPool2D((2, 2)))

model.add(keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(1024, 1024, 3))) # input layer
model.add(keras.layers.MaxPool2D((2, 2)))

model.add(keras.layers.Conv2D(4, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(1024, 1024, 3))) # input layer
model.add(keras.layers.MaxPool2D((2, 2)))




model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax')) # OUTPUT LAYER
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
             optimizer=keras.optimizers.SGD(learning_rate=0.01))
model.fit(training_images_as_numpy, label_as_numpy, epochs=10, batch_size=1)