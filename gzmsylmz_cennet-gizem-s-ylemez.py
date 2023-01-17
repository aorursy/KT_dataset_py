#Burada kütüphanelerimizi tanımdım.

import numpy as np # linear algebra

import keras

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers.convolutional import *

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import random as rn

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2                  

from tqdm import tqdm

print(os.listdir("../input"))
X=[]

Y=[]

img_size=150

shoes= '../input/dl-veriler/shoes'

socks= '../input/dl-veriler/socks'
#Resimlerin tipini dönüştürmek için kullandım.

def assign_label(img,img_type):

    return img_type
#Fotoğrafların yüklenmesi

def make_train_data(img_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,img_type)

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        if img is not None:

            img = cv2.resize(img, (img_size,img_size))

            X.append(np.array(img))

            Y.append(str(label))

        else:

            print("image not loaded")
#Ayakkabıların datasını aldım

make_train_data('shoes',shoes)

print(len(X))
#Çorapların datasını aldım

make_train_data('socks',socks)

print(len(X))
#Veri setinden random olarak görselleştirdim

fig,ax=plt.subplots(5,2)

fig.set_size_inches(17,17)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Y))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Veri: '+Y[l])        

plt.tight_layout()
#Verileri arraya dönüştürme

X=np.array(X) 

X=X.astype('float32') / 255

Y = np.asarray(Y)
#Arrayin içinde kaç farklı değer olduğunu gösteriyoruz

pd.unique(Y)
#Veri setini train data, validation data olarak bölüyoruz

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
#Y dizisi için Label Encoding (i.e. Daisy->0, Rose->1 etc...)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y_test=le.fit_transform(y_test)

y_train=le.fit_transform(y_train)
#Modeli tanımladık, 2 veriyi kıyaslıyacağımız için sigmoid kullandım.

from keras import layers 

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(8, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
y_train_binary=to_categorical(y_train,2)

y_test_binary=to_categorical(y_test,2)
#Modeli fit ediyoruz

history =model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data = (x_test,y_test))

#Acc değerlerini görüyoruz

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



acc
model.save('shoes_socks')
#Epoch değerinin grafiğini çiziyoruz

epochs = range(len(acc))





plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
from keras.layers import Dropout

model = models.Sequential()

model.add(layers.Conv2D(8, (3, 3), activation='relu',input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.5))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.5))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
#Dropout kullanarak modeli fit ettik

history =model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data = (x_test,y_test))
#Dropout kullandığımız da çıkan grafik

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
#Ağırlık regularizasyonunu tanımladık ve modeli oluşturduk

from keras.regularizers import l2

model = models.Sequential()

model.add(layers.Conv2D(8, (3, 3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
#Ağırlık regülarizasyonunu kullanarak modeli fit ettik

history =model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data = (x_test,y_test))
#Ağırlık regularizasyonu kullanarak oluşan grafik

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
#AUGMENTATİON
datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
#Augmentation için modeli tanımlıyoruz

model = models.Sequential()

model.add(layers.Conv2D(8, (3, 3), activation='relu',input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
#Augmentation için modeli fit ettik

history =model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data = (x_test,y_test))
#Verilerimizi arttırdık 

from keras.preprocessing import image

fnames = [os.path.join(socks, fname) for fname in os.listdir(socks)]

img_path = fnames[3]



#Resmi oku ve yeniden boyutlandır

img = image.load_img(img_path, target_size=(150, 150))



# (150, 150, 3) Şeklinde bir Numpy dizisine dönüştürün

x = image.img_to_array(img)



# (1, 150, 150, 3)Yeniden şekillendir

x = x.reshape((1,) + x.shape)



# Aşağıdaki .flow () komutu, rasgele dönüştürülmüş görüntülerin yığınlarını oluşturur.

# Sürekli döneceği için durdurmamız gerek 

i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
#Verilerimizi arttırdıktan sonra çıkan grafiğimiz

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
#validaiton acc ve loss değerleri grafikleri

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

plot_model_history(history)