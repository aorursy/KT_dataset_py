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
import keras
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Conv2D , MaxPooling2D
datos = pd.read_csv('../input/train.csv')
x_train = (datos.iloc[:,1:].values).astype('float32')
y_train = (datos.iloc[:,0].values).astype('int32')
#Leemos datos de test
test = pd.read_csv('../input/test.csv')
#print(test.head())
x_test = test.values.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 28, 28)
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
#Redimensionamos
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
#Normalizacion
x_train = x_train/255
x_test = x_test/255
from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
#Creamos datos de entrenamiento y de validacion
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)
model = Sequential()
model.add = (Conv2D(128, (3,3), input_shape=(28, 28, 1), padding='same', activation = 'relu'))
model.add = (MaxPooling2D(2,2))
model.add = (BatchNormalization())
model.add = (Dropout(0.2))
    
model.add = (Conv2D(128, (3,3),strides =1, padding='valid',activation='relu'))
model.add = (MaxPooling2D(2,2))
model.add = (BatchNormalization())
model.add = (Dropout(0.2))

model.add = (Conv2D(128, (3,3),strides =1, padding='valid',activation='relu'))
model.add = (MaxPooling2D(2,2))
model.add = (BatchNormalization())
model.add = (Dropout(0.2))
    
model.add = (Flatten())
    
model.add = (Dense(10, activation = 'softmax'))

adam = Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
def mostrar(mostrar):
    if mostrar==True:
        print("val_acc")
        plt.plot(h.history['val_acc'], c = 'r')
        plt.show()
        print("val_loss")
        plt.plot(h.history['val_loss'], c = 'r')
        plt.show()
#callbacks
#filepath = '../modelo_1/{acc:.4f}-{val_acc:.4f}-{val_loss:.4f}.keras'
#guardar = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, 
#                              save_best_only=True, save_weights_only=False, mode='max')
#h = model.fit(X_train, y_train, batch_size=128, epochs=10,  validation_data=(X_val, y_val), callbacks=[guardar])
h = model.fit(X_train, y_train, batch_size=128, epochs=10,  validation_data=(X_val, y_val))
