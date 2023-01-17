import pandas as pd

test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

print(test.label.shape)

print(train.shape)

x_train = train.iloc[:,1:]

y_train = train.iloc[:,0]

x_test = test.iloc[:,1:]

y_test = test.iloc[:,0]
import numpy as np

import matplotlib.pyplot as plt

from keras.utils import to_categorical



x_train = x_train.values.reshape(-1, 28,28, 1)

x_test = x_test.values.reshape(-1, 28,28, 1)

#il train set contiene 60000 immagini 28x28

print('Training set shape: ', x_train.shape, y_train.shape)

#il test set contiene 10000 immagini 28x28

print('Testing set shape: ', x_test.shape, y_test.shape)



# Trova gli id corrispondenti alle 10 classi di articoli presenti

classes = np.unique(y_train)

n = len(classes)

print('Number of classes: ', n)# sono 10

print('Classes values: ', classes)# sono interi che vanno da 0 a 9



# Mostra un esempio della prima immagine presente nel train set

plt.figure(figsize=[5,5])



plt.subplot(121)

plt.imshow(x_train[0,:,:,0], cmap='gray')#immagine

plt.title("Class: {}".format(y_train[0]))#classe



# Mostra un esempio della prima immagine presente nel test set

plt.subplot(122)

plt.imshow(x_test[0,:,:,0], cmap='gray')#immagine

plt.title("Class: {}".format(y_test[0]))#classe
#converte i valori delle matrici rappresentati le immagini da int8 to float32, 

#e trasforma i valori dei grigi in valori che variano da 0 a 1 (non più da 0 a 255)

x_train = (x_train.astype('float32')) / 255.

x_test = (x_test.astype('float32')) / 255.



#converte le etichette corrispondenti alle classi in one-hot-vector



y_train_one_hot = to_categorical(y_train)

y_test_one_hot = to_categorical(y_test)



#Mostro un esempio di come sono cambiate le etichette delle classi

print('Original class:', y_train[0])

print('Class converted to one-hot:', y_train_one_hot[0])



#Divide il train set in training set real e validation set, per diminuire overfitting e incrementare le performance

from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=13)

#ora il train set contiene 48000 esempi il validation set 12000, mentre il test set è rimasto a 10000 esempi

x_train.shape,x_valid.shape,y_train.shape,y_valid.shape
import tensorflow as tf

from tensorflow.python.client import device_lib 

device_lib.list_local_devices() # let's list all available computing devices
import keras

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU



batch_size = 64

epochs = 20

num_classes = 10



#neural network architecture

fashion_model = Sequential()



fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))



fashion_model.add(MaxPooling2D((2, 2),padding='same'))



fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))



fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))



fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))

fashion_model.add(LeakyReLU(alpha=0.1))                  

fashion_model.add(Dense(10, activation='softmax'))



#compile the model

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#visualize the layers

fashion_model.summary()



#train the model

with tf.device('/GPU:0'):

    fashion_train = fashion_model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, y_valid))

test_eval = fashion_model.evaluate(x_test, y_test_one_hot, verbose=0)

print('Test loss:', test_eval[0])

print('Test accuracy:', test_eval[1])

#plot the accuracy and loss between training and validation data

accuracy = fashion_train.history['accuracy']

val_accuracy = fashion_train.history['val_accuracy']

loss = fashion_train.history['loss']

val_loss = fashion_train.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.xlabel('epochs')

plt.ylabel('loss function')

plt.title('Training and validation loss')

plt.legend()

plt.show()
batch_size = 64

epochs = 20

num_classes = 10



fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D((2, 2),padding='same'))

fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))                  

fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Dropout(0.4))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))

fashion_model.add(LeakyReLU(alpha=0.1))           

fashion_model.add(Dropout(0.3))

fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.summary()



fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

with tf.device('/GPU:0'):

    fashion_train_dropout = fashion_model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, y_valid))
#plotting performance

test_eval = fashion_model.evaluate(x_test, y_test_one_hot, verbose=1)

print('Test loss:', test_eval[0])

print('Test accuracy:', test_eval[1])



accuracy = fashion_train_dropout.history['accuracy']

val_accuracy = fashion_train_dropout.history['val_accuracy']

loss = fashion_train_dropout.history['loss']

val_loss = fashion_train_dropout.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.title('Training and validation loss')

plt.legend()

plt.show()