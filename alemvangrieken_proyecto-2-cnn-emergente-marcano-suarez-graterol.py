# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd #para leer el dataset en un dataframe

import numpy as np #para calculos de algebra

#librerias para graficas

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline





from sklearn.model_selection import train_test_split #libreria de prepocesamiento de datos

from sklearn.metrics import confusion_matrix #libreria imprimir la matriz de confusion

from sklearn.metrics import roc_auc_score

import itertools

from keras.utils import np_utils



from matplotlib import pyplot #Graficar el Data Augmentation



from keras.utils.np_utils import to_categorical # libreria para el one hot encoding



#librerias para el modelo de la red en Keras

from keras.utils.vis_utils import plot_model #impresion en imagen del modelo

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization



np.random.seed(2) #inicializacion de la semilla

sns.set(style='white', context='notebook', palette='deep')
#Cargar la data en dataframe train y test

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape) #dimensiones del train

print(test.shape) #dimensiones del test

train.head() 
train["label"] #el dataset train incluye el label de que numero es del 0 al 9
#Voy a guardar el label en la variable Y_train

Y_train = train["label"]



# y se la borro a train y lo guardo X_train

X_train = train.drop(labels = ["label"],axis = 1)
#seaborn.countplot Muestra los conteos de observaciones en cada  categoria usando barras.

grafica = sns.countplot(Y_train)
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train / 255.0

test = test / 255.0



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
random_seed=2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
# Arquitectura final





model = Sequential()



#Conv2D->relu ->Conv2D->relu ->MaxPool2D -> Dropout

model.add(Conv2D(filters = 32, strides = (1, 1), kernel_size = (7,7),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, strides = (1, 1), kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.3))



#Conv2D->relu ->Conv2D->relu ->MaxPool2D -> Dropout

model.add(Conv2D(filters = 128, strides = (1, 1), kernel_size = (4,4),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 256, strides = (2, 2), kernel_size = (2,2),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.3))



#-> Flatten -> Dense -> Dropout-> Dense -> Dropout -> Prediccion

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(126, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
model.summary()

model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
datagen = ImageDataGenerator(

        rotation_range=15,  #girar aleatoriamente las imágenes en el rango (grados, 0 a 180)

        zoom_range = 0.15, # Ampliar imagen

        width_shift_range=0.1,  # cambiar aleatoriamente las imágenes horizontalmente 

        height_shift_range=0.1) # cambiar aleatoriamente las imágenes verticalmente 

        #otra prueba que hicimos

    #rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)

datagen.fit(X_train)


#Impresion de la configuracion en el dataGenerator de forma random se escogen las observaciones pero se muestran siempre 9

for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=9): #se le pasa la configuracion anterior del datagen

   

    for i in range(0, 9):

        pyplot.subplot(330 + 1 + i)

        pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

    pyplot.show()

    break
batch_size=32

epochs=36





history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), validation_data = (X_val,Y_val),

                              steps_per_epoch=X_train.shape[0] // batch_size, epochs = epochs

                              , callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Entrenamiento loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validacion loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Entrenamiento accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validacion accuracy")

legend = ax[1].legend(loc='best', shadow=True)
_, train_acc = model.evaluate(X_train, Y_train, verbose=0)

_, test_acc = model.evaluate(X_val, Y_val, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
#Imprimir la matriz mas grande

plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Complexity Graph:  Training vs. Validation Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validate'], loc='upper right')



plt.figure(2)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy Graph:  Training vs. Validation accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validate'], loc='upper right')

plt.show()
#funcion que imprime la matriz de confusion

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

  

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('Valor Verdadero')

    plt.xlabel('Valor Predicho')



# Predecir los valores del set de validación

Y_pred = model.predict(X_val)

#Convertir la prediccion a vectores one hot (categoricas)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

#Convertir las observaciones a vectores one hot (categoricas)

Y_true = np.argmax(Y_val,axis = 1) 

#desplegar matriz

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# graficar matriz

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
result = model.predict(test)

result = np.argmax(result, 1)

predictions = result.T

result = pd.DataFrame({'ImageId': range(1,len(predictions)+1), 'Label': predictions})

result.to_csv('result.csv', index=False, encoding='utf-8')