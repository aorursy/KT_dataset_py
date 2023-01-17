# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# importation des librairies et packages Kears 
from keras.models import Sequential # il existe deux manieres d'initaliser un réseau, soit en séquence de couche ou en graphe
from keras.layers import Conv2D # this is to perform the convolution operation i.e the first step of a CNN (3D in case of video)
from keras.layers import MaxPooling2D # MaxPooling we need the maximum value pixel from the respective region of interest.
from keras.layers import Flatten #  Flattening is the process of converting all the resultant 2 dimensional arrays into a single long continuous linear vector.
from keras.layers import Dense # is used to perform the full connection of the neural network
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib.pyplot import imshow
from skimage.transform import resize
print( os.listdir("../input/test_set-20180919t120343z-001/test_set/"))

# Any results you write to the current directory are saved as output.
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 # Ce modele m'a générer une erreur 
from keras.applications.vgg16 import decode_predictions
from tabulate import tabulate
#model = VGG16(weights = 'imagenet', include_top = False) # Création du modèle VGG-16 implementé par Keras
#img =np.asarray( load_img("../input/test_set-20180919t120343z-001/test_set/cats/cat.4001.jpg",target_size = (244,244)))
#img = img_to_array(img)
#img = img.reshape((1,img.shape[0], img.shape[1], img.shape[2] ))
#img = preprocess_input(img)
# Afficher les 3 classes les plus probables
#print('Top 3 :', decode_predictions(y, top=3)[0])
from keras.applications.mobilenet import MobileNet, decode_predictions
from keras.models import load_model
model = MobileNet()
#model = VGG16(weights = 'imagenet')
img =np.asarray( load_img("../input/test_set-20180919t120343z-001/test_set/cats/cat.4001.jpg",target_size = (224,224)))
img = resize(img, [224, 224])
imshow(img)  # affichage de l'image
X = np.reshape(img, [1, 224, 224, 3])
y = model.predict(X)
print(tabulate(decode_predictions(y, top=5)[0], headers=['Name', 'Probability']))


from keras.applications import VGG16
from keras.layers import Dense
# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# Récupérer la sortie de ce réseau
x = model.output
print (x)
model.summary()
# Ajouter la nouvelle couche fully-connected pour la classification à 10 classes
predictions = Dense(10, activation='softmax')(x)
# Définir le nouveau modèle
new_model = Model(inputs=model.input, outputs=predictions)
 #1 : fine-tuning total
#for layer in model.layers:
 #   layer.trainable = True
#2 : extraction de features
#for layer in model.layers:
#    layer.trainable = False
#3 : fine-tuning partiel
# Ne pas entraîner les 5 premières couches (les plus basses) 
for layer in model.layers[:5]:
    layer.trainable = False
# Compiler le modèle 
new_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
# Entraîner sur les données d'entraînement (X_train, y_train)
model_info = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)



from keras.applications.resnet50 import resnet50,decode_predictions
base_model = resnet50 (weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 7 classes
predictions = Dense(7, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)