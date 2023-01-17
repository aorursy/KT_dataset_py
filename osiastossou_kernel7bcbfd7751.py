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
os.listdir('/kaggle/input')
import numpy as np

import os

import cv2

from PIL import Image

import matplotlib.pyplot as plt
from keras.datasets import mnist

from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input

dossier_train = os.listdir('/kaggle/input/10-monkey-species/training/training/')

x = np.array([])

y = []

i = 0

u = 0

for dossier in dossier_train:

    all_names_img = os.listdir('/kaggle/input/10-monkey-species/training/training/'+dossier)

    for name in all_names_img:

        img = load_img('/kaggle/input/10-monkey-species/training/training/'+dossier+'/'+name, target_size=(150, 150,3))  # Charger l'image

        img = img_to_array(img)  # Convertir en tableau numpy

        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)

        img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

        if u == 0:

            x = img

            u = 1

        else:

            x = np.append(x,img,axis=0)

        y.append(i)

    i=i+1

y = np.array(y)
plt.imshow(x[2])
from sklearn.model_selection import train_test_split

x_train, x_test, y_train_1, y_test_1 = train_test_split(x, y, test_size=0.33, random_state=12)



from keras.utils import to_categorical

y_train = to_categorical(y_train_1)

y_test = to_categorical(y_test_1)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.layers import Dense

from keras import Model



# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected

model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3),pooling="max")

#model_vgg19 = VGG19(weights="imagenet")

# Récupérer la sortie de ce réseau

#x = model.output



# Ajouter la nouvelle couche fully-connected pour la classification à 10 classes

#predictions = Dense(10, activation='softmax')(x)



# Définir le nouveau modèle



#new_model = Model(inputs=model.input, outputs=predictions)
for layer in model.layers:

    layer.trainable = False

    

from keras import optimizers

epochs = 10

batch_size = 50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.models import Model, Sequential



# Although this part can be done also with the functional API, I found that for this simple models, this becomes more intuitive

new_model = Sequential()

for layer in model.layers:

    new_model.add(layer)

new_model.add(Dense(100, activation="relu"))  # Very important to use relu as activation function, search for "vanishing gradiends" :)

new_model.add(Dropout(0.5))

new_model.add(Dense(3, activation="softmax")) 

# Compiler le modèle 

new_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
from keras.callbacks import LearningRateScheduler

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = new_model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=20, #Increase this when not on Kaggle kernel

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(x_test, y_test), #For speed

                           callbacks=[annealer])