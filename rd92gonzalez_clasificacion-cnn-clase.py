from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Modelo de red convolucional

#Estructura base
cnn = Sequential()

#Primera capa convolucional
cnn.add(Conv2D(filters=32, kernel_size = (3,3), input_shape = (64,64,3), activation = "relu"))

#Maxpooling
cnn.add(MaxPooling2D(pool_size=(4,4)))

#Se pueden seguir agregando capas
#Segunda capa convolucional
#cnn.add(Conv2D(filters=32, kernel_size = (3,3), input_shape = (64,64,3), activation = "relu"))

#Maxpooling
#cnn.add(MaxPooling2D(pool_size=(4,4)))

#Capa de flattering
cnn.add(Flatten())

#red neuronal -> Fully connected
cnn.add(Dense(units = 128, activation = "relu"))

#capa de salida, para clasificacion binaria
cnn.add(Dense(units = 1, activation="sigmoid"))
#Compilar la red
cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
strDataTrain = "../input/cat-and-dog/training_set/training_set"
strDataTest = "../input/cat-and-dog/test_set/test_set"
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip = True

)

#Preprocesamiento para test
test_datagen = ImageDataGenerator(
    rescale = 1./255.

)
#Configuracion de imagenes de entrada para train
train_set = train_datagen.flow_from_directory(
    strDataTrain,
    target_size = (64,64),
    batch_size = 32,
    class_mode ="binary"
)

#Configuracion de imagenes de entrada para test
test_set = test_datagen.flow_from_directory(
    strDataTest,
    target_size = (64,64),
    batch_size = 32,
    class_mode ="binary"
)
#cnn.fit(train_set, steps_per_epoch = 8000, epochs = 3, validation_data = test_set,
#        validation_steps = 2000, verbose = 2)
cnn.fit(train_set, epochs = 10, validation_data = test_set, verbose = 2)
cnn.save('PrimerModelo.h5')
type(train_set)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# Funcion que debe devolver nuestra red neuronal. Aqui se le llenan con parametros dinamicos segun lo que se necesite
def create_model(optimizer, init_mode, activation, neurons, filter1, filter2, padding, strides, activationLast):
    #Estructura base
    cnn = Sequential()

    #Primera capa convolucional
    cnn.add(Conv2D(filters=filter1, kernel_size = (3,3), input_shape = (64,64,3), activation = activation, padding = padding))

    #Maxpooling
    cnn.add(MaxPooling2D(pool_size=(4,4), strides = strides))

    #Se pueden seguir agregando capas
    #Segunda capa convolucional
    cnn.add(Conv2D(filters=filter2, kernel_size = (3,3), activation = activation, padding = padding))

    #Maxpooling
    cnn.add(MaxPooling2D(pool_size=(4,4), strides = strides))

    #Capa de flattering
    cnn.add(Flatten())

    #red neuronal -> Fully connected
    cnn.add(Dense(units = neurons, kernel_initializer=init_mode, activation = activation))

    #capa de salida, para clasificacion binaria
    cnn.add(Dense(units = 1, activation=activationLast))
    
    #Compilar la red
    cnn.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    
    
    
    return cnn
modeloFinal = create_model(optimizer = 'adam', init_mode='glorot_uniform', activation = 'relu', neurons= 20, 
                          filter1 = 32, filter2 = 64, padding = 'same', strides = 2, activationLast = 'sigmoid')
modeloFinal.fit(train_set, epochs = 10, validation_data = test_set, verbose = 2)
modeloFinal.save('SegundoModelo.h5')
modeloFinal = create_model(optimizer = 'adam', init_mode='glorot_uniform', activation = 'relu', neurons= 40, 
                          filter1 = 32, filter2 = 64, padding = 'same', strides = 2, activationLast = 'sigmoid')
modeloFinal.fit(train_set, epochs = 10, validation_data = test_set, verbose = 2)
modeloFinal.save('TercerModelo.h5')
modeloFinal = create_model(optimizer = 'adam', init_mode='glorot_uniform', activation = 'relu', neurons= 40, 
                          filter1 = 64, filter2 = 64, padding = 'same', strides = 2, activationLast = 'sigmoid')
modeloFinal.fit(train_set, epochs = 10, validation_data = test_set, verbose = 1)
modeloFinal.save('CuartoModelo.h5')
train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip = True

)

#Preprocesamiento para test
test_datagen = ImageDataGenerator(
    rescale = 1./255.

)
#Configuracion de imagenes de entrada para train
train_set = train_datagen.flow_from_directory(
    strDataTrain,
    target_size = (128,128),
    batch_size = 32,
    class_mode ="binary"
)

#Configuracion de imagenes de entrada para test
test_set = test_datagen.flow_from_directory(
    strDataTest,
    target_size = (128,128),
    batch_size = 32,
    class_mode ="binary"
)
# Funcion que debe devolver nuestra red neuronal. el shape se le agregara como variable de entrada
def create_model2(optimizer, init_mode, activation, neurons, filter1, filter2, padding, strides, activationLast, input_shape):
    #Estructura base
    cnn = Sequential()

    #Primera capa convolucional
    cnn.add(Conv2D(filters=filter1, kernel_size = (3,3), input_shape = input_shape, activation = activation, padding = padding))

    #Maxpooling
    cnn.add(MaxPooling2D(pool_size=(4,4), strides = strides))

    #Se pueden seguir agregando capas
    #Segunda capa convolucional
    cnn.add(Conv2D(filters=filter2, kernel_size = (3,3), activation = activation, padding = padding))

    #Maxpooling
    cnn.add(MaxPooling2D(pool_size=(4,4), strides = strides))

    #Capa de flattering
    cnn.add(Flatten())

    #red neuronal -> Fully connected
    cnn.add(Dense(units = neurons, kernel_initializer=init_mode, activation = activation))

    #capa de salida, para clasificacion binaria
    cnn.add(Dense(units = 1, activation=activationLast))
    
    #Compilar la red
    cnn.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    
    
    
    return cnn
modeloFinal = create_model2(optimizer = 'adam', init_mode='glorot_uniform', activation = 'relu', neurons= 60, 
                          filter1 = 64, filter2 = 64, padding = 'same', strides = 2, activationLast = 'sigmoid', input_shape = (128, 128, 3))
modeloFinal.fit(train_set, epochs = 10, validation_data = test_set, verbose = 1)
modeloFinal.save('QuintoModelo.h5')
from keras.models import load_model
Salida = load_model('TercerModelo.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/pruebaperro/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = Salida.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)
import cv2
import matplotlib.pyplot as plt
def PlotImagen(nombre, grises = True):
    img = cv2.imread(nombre)
    if grises == True:
        #img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show
#Imagen a usar
PlotImagen('../input/pruebaperro/cat_or_dog_1.jpg')
test_image = image.load_img('../input/pruebaperro/cat.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = Salida.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)
#Imagen a usar
PlotImagen('../input/pruebaperro/cat.jpg')