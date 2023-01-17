import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle



import matplotlib.pylab as plt

import time

import json

import keras

from keras.models import Model, Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D

from keras.layers import Input, Concatenate, Flatten, UpSampling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras.preprocessing.image import load_img

import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.utils import np_utils

from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

from keras_applications.resnext import ResNeXt50

from keras_applications.resnet import ResNet50

from keras.initializers import VarianceScaling

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from keras.models import load_model



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        

        fl = os.path.join(dirname, filename)

        #if fl != '/kaggle/working/__notebook_source__.ipynb':

        print(os.path.join(dirname, filename))

            #os.remove(fl)



# Any results you write to the current directory are saved as output.
def get_data(trainPath, testPath):

    """Cette fonction retourne les données d'apprentissage et de text

    params:

        ---> trainPath : chemin de la directory des images d'apprentissage

        ---> trainPath : chemin de la directory des images de test

    retour :

        ---> trainGen : générateur d'image d'apprentissage

        ---> trainGen : générateur d'image de test

        ---> train_x : tableau d'image d'apprentissage

        ---> train_y: tableau des classes d'apprentissage

        ---> test_x : tableau d'image de test

        ---> test_y : tableau des classes de test

    """

    

    # instancier un objet ImageDataGenerator pou l'augmentation des donnees train

    trainAug = ImageDataGenerator(rescale = 1./255, horizontal_flip=True,fill_mode="nearest")

    testAug = ImageDataGenerator(rescale = 1./255)

    

    # definir la moyenne des images ImageNet par plan RGB pour normaliser les images de la base AFF20

    mean = np.array([123.68, 116.779, 103.939], dtype="float32")/255

    trainAug.mean = mean

    testAug.mean = mean



    # initialiser le generateur de train

    trainGen = trainAug.flow_from_directory(

    trainPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=True,

    batch_size=16)



    # initialiser le generateur de test

    testGen = testAug.flow_from_directory(

    testPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=False,

    batch_size=16)

    

    

    #Lire les données sous forme de tableaux numpy, pour l'évalusation

    #puisque la fonction fit de la class gridsearchcv prend en paramétre des

    #tableaux et non pas des générateur.

    

    #pour cette partie on peut bien lire la base de données manuelement (des boucle for)

    #mais dans ce cas on fera l'évaluation avec des données non augmenter, et l'apprentissage

    #avec des données augmenter. pour cela on extrait les tableaux à partir des générateur eux même.

    #c'est aussi plus rapide que d'utiliser des boucles.

    

    #les dimension des deux bases

    n_train = trainGen.samples

    n_test = testGen.samples

    

    # initialiser le generateur de train

    trainGen_tmp = trainAug.flow_from_directory(

    trainPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=True,

    batch_size=n_train)



    # initialiser le generateur de test

    testGen_tmp = testAug.flow_from_directory(

    testPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=False,

    batch_size=n_test)

    

    

    train_x = trainGen_tmp.next()[0]

    train_y = trainGen_tmp.next()[1]

    

    test_x = testGen_tmp.next()[0]

    test_y = testGen_tmp.next()[1]

    

    print("x_train_shape:",train_x.shape)

    print("y_train_shape:",test_y.shape)

    

    print("x_test_shape:",test_x.shape)

    print("y_test_shape:",test_y.shape)

    

    return trainGen,testGen, train_x, train_y, test_x, test_y

trainPath = "/kaggle/input/db-inf907/AFF20_crops/AFF20_crops/AFF20_crops_train/"

testPath = "/kaggle/input/db-inf907/AFF20_crops/AFF20_crops/AFF20_crops_test/"



trainGen,testGen, train_x, train_y, test_x, test_y = get_data(trainPath, testPath)

# Un réseau en couches séquentielles

Auto_encoder = Sequential()



# Premier niveau de convolution - pooling

Auto_encoder.add(Conv2D(16,(3, 3),padding = 'Same',activation="relu", input_shape=(224, 224, 3) ,kernel_initializer="he_uniform"))

Auto_encoder.add(MaxPooling2D(pool_size = (2, 2), padding='same'))



# Deuxième niveau de convolution - pooling

Auto_encoder.add(Conv2D(32, (3, 3), activation="relu" , padding='same' ,kernel_initializer="he_uniform"))

Auto_encoder.add(MaxPooling2D(pool_size = (2, 2), padding='same'))



# Troisième niveau de convolution - pooling

Auto_encoder.add(Conv2D(64, (3, 3), activation="relu" , padding='same',kernel_initializer="he_uniform"))

Auto_encoder.add(MaxPooling2D(pool_size = (2, 2), padding='same'))



# Quatreième niveau de convolution - pooling

Auto_encoder.add(Conv2D(128, (3, 3), activation="relu" , padding='same' ,kernel_initializer="he_uniform"))

Auto_encoder.add(MaxPooling2D(pool_size = (2, 2), padding='same'))



# Premier niveau de deconvolution  - UpSampling2D

Auto_encoder.add(Conv2D(128, (3, 3), activation="relu" , padding='same',kernel_initializer="he_uniform"))

Auto_encoder.add(UpSampling2D((2, 2)))



# Premier niveau de deconvolution  - UpSampling2D

Auto_encoder.add(Conv2D(64, (3, 3), activation="relu" , padding='same',kernel_initializer="he_uniform"))

Auto_encoder.add(UpSampling2D((2, 2)))



# Deuxième niveau de deconvolution  - UpSampling2D

Auto_encoder.add(Conv2D(32, (3, 3), activation="relu" , padding='same',kernel_initializer="he_uniform"))

Auto_encoder.add(UpSampling2D((2, 2)))



# Troisième niveau de deconvolution - UpSampling2D

Auto_encoder.add(Conv2D(16, (3, 3), activation="relu" , padding='same',kernel_initializer="he_uniform"))

Auto_encoder.add(UpSampling2D((2, 2)))



# Dérnier convolution pour arrivé à la taille de l'image initial (244,244,3)

Auto_encoder.add(Conv2D(3, (3, 3), activation='relu', padding='same',kernel_initializer="he_uniform"))



# Compilation du CNN décrit

Auto_encoder.compile(

    optimizer = 'adadelta',

    loss = 'MSE',

    metrics = ['accuracy'])



Auto_encoder.summary()
# instancier un objet ImageDataGenerator pou l'augmentation des donnees train pour l'auto-encodeur

trainAug = ImageDataGenerator(horizontal_flip=True,fill_mode="nearest")

testAug = ImageDataGenerator()



# definir la moyenne des images ImageNet par plan RGB pour normaliser les images de la base AFF20

mean = np.array([123.68, 116.779, 103.939], dtype="float32")

trainAug.mean = mean

testAug.mean = mean



# initialiser le generateur de train

trainGen_autoencod = trainAug.flow_from_directory(

    trainPath,

    class_mode="input",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=True,

    batch_size=64)



# initialiser le generateur de test

testGen_autoencod = testAug.flow_from_directory(

    testPath,

    class_mode="input",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=False,

    batch_size=64)
# Entrainement du Réseau

epochs= 200 

batch_size=64



Auto_encoder.fit(train_x,train_x,batch_size=batch_size,epochs=epochs)
predicted_test_img = Auto_encoder.predict(test_x)
n=9

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i+1)

    plt.imshow(test_x[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + n+1)

    plt.imshow(predicted_test_img[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
Auto_encoder.save("Auto_encoder")
def extract_carts(model_path,X):

    """Cette fonction extraire les vecteurs descripteurs des images

    """

    model = load_model(model_path)

    

    model2 = Model(input=model.input, output=[model.layers[7].output])

    

    vec_cara = model2.predict(X)

    

    return vec_cara

#extraire les vecteurs et les sauvegarder

model_path = "/kaggle/working/Auto_encoder"

vec_cara = extract_carts(model_path,train_x)

pd.DataFrame(vec_cara).to_csv('Imgs_Vectors.csv')