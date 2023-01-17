import numpy as np

from matplotlib import pyplot as plt

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os

import shutil

import multiprocessing as mp

import cv2

import numpy as np

import sklearn.metrics as metrics

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.models import load_model

from keras.models import Sequential

from keras import optimizers

from keras.utils import to_categorical

from sklearn.utils import class_weight

from keras import layers

from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D, MaxPooling2D, Conv2D, Input

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.models import Model

from keras.optimizers import SGD







from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50



%matplotlib inline



train_dir = '/kaggle/input/food11/training'

validation_dir = '/kaggle/input/food11/validation'

train_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]

validation_files = [f for f in os.listdir(validation_dir) if os.path.isfile(os.path.join(validation_dir, f))]
from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



# Extracci'on de labels

train = []

y_train = []

valid = []

y_valid = []



for file in train_files:

    train.append(file)

    label= file.find("_")

    y_train.append(int(file[0:label]))

for file in validation_files:

    valid.append(file)

    label= file.find("_")

    y_valid.append(int(file[0:label]))
cnnInput = np.ndarray(shape=(len(train), 190,190, 3), dtype=np.float32)

i=0

for file in train:

    image = cv2.imread(train_dir + "/" + file)  

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype("float")

    image = cv2.resize(image, dsize=(190, 190), interpolation=cv2.INTER_CUBIC)

    x = img_to_array(image)

    x = x.reshape((1, x.shape[0], x.shape[1],

                                   x.shape[2]))



    cnnInput[i]=x

    i+=1
cnnValidation = np.ndarray(shape=(len(valid), 190,190, 3), dtype=np.float32)

i=0

for file in valid:

    image = cv2.imread(validation_dir + "/" + file)  

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype("float")

    image = cv2.resize(image, dsize=(190, 190), interpolation=cv2.INTER_CUBIC)



    x = img_to_array(image)

    x = x.reshape((1, x.shape[0], x.shape[1],

                                   x.shape[2]))



    cnnValidation[i]=x

    i+=1

y_train_2 = to_categorical(y_train)

y_valid_2 = to_categorical(y_valid)
vgg_model = VGG19(weights='imagenet', include_top=False)
# Se crea las variables hot-encoded

#hot-encoded-> representaci'on de variables categ'oricas como vectores binarios

y_train_hot_encoded = to_categorical(y_train)

y_test_hot_encoded = to_categorical(y_valid)
class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)
#se obtienen las capas y se agrega una capa de GlobalAveragePooling

x = vgg_model.output

x = GlobalAveragePooling2D()(x)



# se agrega una capa fully-connected 

x = Dense(2048, activation='relu')(x)

x = Dropout(0.3)(x)



# Se agregan las capas de salida

predictions = Dense(11, activation='softmax')(x)



model = Model(inputs=vgg_model.input, outputs=predictions)

#congelar el las capas del modelo preentrenado

for layer in vgg_model.layers:

    layer.trainable = False



# Actualizaci'on de los pesos que son agragados

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(cnnInput, y_train_hot_encoded)



# Se eligen las capas que seran actualizadas durante el entrenamiento

layer_num = len(model.layers)

for layer in model.layers[:21]:

    layer.trainable = False



for layer in model.layers[21:]:

    layer.trainable = True



# Entrenamiento

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

#parametro que acelera el sgd en la direcci'on relevante, y amortigua las oscilaciones

#SDG stocastic gradient descent optimizer-> merodo iterativo para optimizar una funcion objetivo, con propiedades de suavidad adecuada

#history= model.fit(cnnInput,y_train_hot_encoded, batch_size=64, shuffle=True,

#                    validation_data=(cnnValidation, y_test_hot_encoded),

#                  class_weight=class_weights, epochs=100)

#history = model.fit(cnnInput, y_train_hot_encoded, batch_size=256, epochs=50, shuffle=True,  validation_split=0.1)
model.summary()
# Data augmentation

from keras.preprocessing.image import ImageDataGenerator

# Configuraci'on de data augmentation

train_datagen = ImageDataGenerator(

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=[.6, 1],

    vertical_flip=True,

    horizontal_flip=True)

train_generator = train_datagen.flow(cnnInput, y_train, batch_size=64, seed=11)

valid_datagen = ImageDataGenerator()

valid_generator = valid_datagen.flow(cnnValidation, y_valid, batch_size=64, seed=11)
train_datagen.fit(cnnInput)

valid_datagen.fit(cnnValidation)
hystory=model.fit_generator(train_datagen.flow(cnnInput, y_train_hot_encoded, batch_size=64), shuffle=True,

                    validation_data=valid_datagen.flow(cnnValidation, y_test_hot_encoded, batch_size=64),

                  class_weight=class_weights, epochs=300)
model.summary()
model.save('modelo300.h5')

model.save_weights('pesos300.h5')
import matplotlib.pyplot as plt

import numpy as np
plt.figure(0)  

plt.plot(hystory.history['accuracy'],'r')  

plt.plot(hystory.history['val_accuracy'],'g')  

plt.xticks(np.arange(0, 301, 25.0))  

plt.rcParams['figure.figsize'] = (12,10 )  

plt.xlabel("Num of Epochs")  

plt.ylabel("Accuracy")  

plt.title("Training Accuracy vs Validation Accuracy")  

plt.legend(['train','validation'])



plt.figure(1)  

plt.plot(hystory.history['loss'],'r')  

plt.plot(hystory.history['val_loss'],'g')  

plt.xticks(np.arange(0, 301, 25.0))  

plt.rcParams['figure.figsize'] = (12, 10)  

plt.xlabel("Num of Epochs")  

plt.ylabel("Loss")  

plt.title("Training Loss vs Validation Loss")  

plt.legend(['train','validation'])



plt.show() 


import keras

from keras.preprocessing.image import load_img, img_to_array

from keras.models import load_model

import tensorflow as tf 

longitud, altura = 190, 190#tamaño de la imagen

#modelo = 'modelo.h5'#direccion del modelo

#pesos_modelo = 'pesos.h5'#direccion de los pesos

#model = tf.keras.models.load_model(modelo)#cargar modelo

#model.load_weights(pesos_modelo)#cargar pesos

#funcion de prediccion

def predict(file):

  x = tf.keras.preprocessing.image.load_img(file, target_size=(longitud, altura))#cargar la imagen

  x = tf.keras.preprocessing.image.img_to_array(x)#tansformar imagen a arreglo

  x = np.expand_dims(x, axis=0)#en el eje 0 se agrega una dimención extra para procesar la información sin problema

  array = model.predict(x)#se llama a la red para realizar la predicción

  result = array[0]#obtenemos el resultado

  answer = np.argmax(result)#nos entrega el indice del valor mas alto

  #clasificación del resultado

  if answer == 0:

    print(file+" pred: Pan 0")

  elif answer == 1:

    print(file+" pred: Lacteo 1")

  elif answer == 2:

    print(file+" pred: Postre 2")

  elif answer == 3:

    print(file+" pred: Huevos 3")

  elif answer == 4:

    print(file+" pred: Fritura 4")

  elif answer == 5:

    print(file+" pred: Carne 5")

  elif answer == 6:

    print(file+" pred: Pasta 6")

  elif answer == 7:

    print(file+" pred: Arroz 7")

  elif answer == 8:

    print(file+" pred: Marisco 8")

  elif answer == 9:

    print(file+" pred: Sopa 9")

  elif answer == 10:

    print(file+" pred: Fruta-Verdura 10")

  return answer
from os import scandir, getcwd

def ls(ruta = getcwd()):

    return [arch.name for arch in scandir(ruta) if arch.is_file()]
lista_arq = ls('/kaggle/input/food11/evaluation') 

true=0

for i in lista_arq:

   p=predict('/kaggle/input/food11/evaluation/'+i)

   if(str(p)==i[0]):

     true=true+1

print(100*true/len(lista_arq))