# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import keras

from keras.datasets import mnist

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, MaxPooling2D

import matplotlib.pyplot as plt

# Download the MNIST dataset and partition train / test

train = pd.read_csv("../input/digit-recognizer/train.csv") 

test = pd.read_csv("../input/digit-recognizer/test.csv") 



x_train_orig = np.array(train.iloc[:, :-1].values)

y_train_orig = np.array(train.iloc[:, 1].values)

x_test_orig = np.array(test.iloc[:, :-1].values)

y_test_orig = np.array(test.iloc[:, 1].values)

# Checking an example

first_image = x_train_orig[0]

first_image = np.array(first_image, dtype='float')

pixels = first_image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()
# Shape

x_train_orig[0].shape
print('Training data shape : ', x_train_orig.shape, y_train_orig.shape)



print('Testing data shape : ', x_test_orig.shape, y_test_orig.shape)

nRows, nCols, nDims = x_train_orig.shape

input_shape = (nRows, nCols, nDims)

x_train = x_train_orig.reshape(60000, nCols, nDims, 1)

x_test = x_test_orig.reshape(10000, nCols, nDims, 1)
print(nRows, nCols, nDims, 1)
x_train = x_train / 255

x_test = x_test / 255



num_classes = 10

y_train = to_categorical(y_train_orig, num_classes)

y_test = to_categorical(y_test_orig, num_classes)
y_train[5]
num_classes = 10



model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Entrenamiento del modelo

n_epochs_onelayer = 9

mfit_onelayer = model.fit(x_train, y_train,

                              epochs=n_epochs_onelayer,

                              verbose=1,

                              validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# plot del training loss y el accuracy

def plot_prediction(n_epochs, mfit):



    # Plot training & validation loss values

    plt.plot(mfit.history['loss'])

    plt.plot(mfit.history['val_loss'])

    title = 'Model loss (' + str(n_epochs) + ' epochs)'

    plt.title(title)

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    

    return plt
plot_prediction(n_epochs_onelayer, mfit_onelayer)
mfit_onelayer
# Hacemos la predicción para las 4 primeras imágenes del set de test

print(model.predict(x_test[:4]))



# Mostramos el ground truth para las primeras 4 imágenes

y_test[:4]
y_train = to_categorical(y_train_orig)

y_test = to_categorical(y_test_orig)
def creaModeloRedNeuronalProfunda():

    num_classes = 10



    model = Sequential()



    ##TODO: Añadir las capas

    model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    

    return model

batch_size = 128

n_epochs = 12 



deep_model = creaModeloRedNeuronalProfunda()



deep_model.compile(optimizer='adadelta',

                   loss='categorical_crossentropy',

                   metrics=['accuracy'])



deep_model.summary()
mfit = deep_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, 

                   validation_data=(x_test, y_test))





# Evaluation in test

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

# Visualizción de la evolución de la métrica accuracy

plot_prediction(n_epochs, mfit)
# Prediction

print(model.predict(x_test[:4]))

y_test[:4]
x_train = x_train_orig[:, ::2, ::2]

x_test = x_test_orig[:, ::2, ::2]
# Normalizamos los valores de los píxeles

x_train = x_train / 255.0

x_test = x_test / 255.0



y_train = x_train_orig / 255.0

y_test = x_test_orig / 255.0



# Ajustamos las dimensiones de los datos

x = np.expand_dims(x_train, axis=3)

y = np.expand_dims(y_train, axis=3)



x_test_final = np.expand_dims(x_test, axis=3)

y_test_final = np.expand_dims(y_test, axis=3)
y_test.shape

y.shape
def creaModeloRedNeuronalConvolucional():

    num_classes = 10



    model = Sequential()



    # Add layer  

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(14, 14, 1)))    

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    # deconvolutional layer

    model.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))

    model.add(keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid'))

    

    return model
# Model creation

convolutional_network = creaModeloRedNeuronalConvolucional()



convolutional_network.summary()
#Entrenar y compilar el modelo

convolutional_network.compile(loss='binary_crossentropy',

              optimizer='Adadelta',

              metrics=['accuracy'])
n_epochs = 12 

convolutional_network.fit(x, y, epochs=n_epochs, validation_data=(x_test_final, y_test_final))
# Predicción de tres imágenes del conjunto de test

n_images = 3

idx_images = np.random.randint(x_test.shape[0], size=n_images)



images = np.expand_dims(np.stack([x_test[i] for i in idx_images]), axis=3)



pred = convolutional_network.predict(images)

# Ajustamos las dimensiones de las imágenes predichas al formato adecuado y desnormalizamos los píxeles

pred = np.squeeze(pred)

pred = pred * 255.0
def drawImage(i, pred, alto, ancho):

  #Visualizamos la imagen i del dataset

  first_image = pred[i]

  first_image = np.array(first_image, dtype='float')

  pixels = first_image.reshape((alto, ancho))

  plt.imshow(pixels, cmap='gray')

  plt.show()
#Visualizamos la primera imagen 

drawImage(0, images, 14, 14)

drawImage(0, pred, 28, 28)

#Visualizamos la segunda imagen 

drawImage(1, images, 14, 14)

drawImage(1, pred, 28, 28)

#Visualizamos la tercera imagen 

drawImage(2, images, 14, 14)

drawImage(2, pred, 28, 28)