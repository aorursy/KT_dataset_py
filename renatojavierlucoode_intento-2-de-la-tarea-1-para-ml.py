# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
data_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
data_train['label'].unique()
from sklearn.model_selection import train_test_split

data_train, data_val = train_test_split(data_train,train_size=0.8)

data_val.info()
data_train.info()
data_train['label'].replace(to_replace=[0,1,2,3,4,5,6,7,8,9],value=[0,0,0,1,0,0,0,0,0,0],inplace =True)
data_train['label'].unique()

data_val['label'].replace(to_replace=[0,1,2,3,4,5,6,7,8,9],value=[0,0,0,1,0,0,0,0,0,0],inplace =True)
data_val['label'].unique()

data_test['label'].replace(to_replace=[0,1,2,3,4,5,6,7,8,9],value=[0,0,0,1,0,0,0,0,0,0],inplace =True)
data_test['label'].unique()
x_train = data_train.iloc[:,1:].values.astype('float32')
y_train = data_train.iloc[:,0].values.astype('int32')

x_val = data_val.iloc[:,1:].values.astype('float32')
y_val = data_val.iloc[:,0].values.astype('int32')

x_test = data_test.iloc[:,1:].values.astype('float32')
y_test = data_test.iloc[:,0].values.astype('int32')
print("Datos para entrenar: ", x_train.shape[0])
print("Datos para validar: ",  x_val.shape[0])
print("Datos para probar: ",   x_test.shape[0])
plt.figure(figsize=(20,20))
for i in range(40):
    plt.subplot(8,5,i+1)
    plt.imshow(np.reshape(x_val[i],(28,28)),cmap='gray')
    plt.title("Clase:{}".format(y_val[i]))
    plt.xticks([])
    plt.yticks([])
x_train = x_train/255

x_val = x_val/255

x_test = x_test/255
from keras.utils import np_utils
num_clases = 2
Y_train = np_utils.to_categorical(y_train,num_clases)
Y_val = np_utils.to_categorical(y_val,num_clases)
Y_test = np_utils.to_categorical(y_test,num_clases)
print(y_train.shape, " ",Y_train.shape)
print(y_val.shape, " ",Y_val.shape)
print(y_test.shape, " ",Y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import activations, initializers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

Epocas = 100


MLP = Sequential()

#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)

#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))

#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

optim = SGD(lr=0.1)

MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])


Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val)
                  )
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(20,20))
for i,nice in enumerate(Correctos[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(x_test[nice].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
              "Predict: {}, Real: {}".format(resultado[nice],
                                            y_test[nice]))
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(20,20))
for i,nice in enumerate(Incorrectos[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(x_test[nice].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
              "Predict: {}, Real: {}".format(resultado[nice],
                                            y_test[nice]))
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
from keras.losses import MeanSquaredError

MLP = Sequential()

#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)

#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))

#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

optim = SGD(lr=0.1)

MLP.compile(loss=MeanSquaredError(), optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val)
                  )
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100
MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

optim = SGD(lr=0.01)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(20,20))
for i,nice in enumerate(Correctos[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(x_test[nice].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
              "Predict:{}, Real:{}".format(resultado[nice],
                                            y_test[nice]))
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(20,20))
for i,nice in enumerate(Incorrectos[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(x_test[nice].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
              "Predict:{}, Real:{}".format(resultado[nice],
                                            y_test[nice]))
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

optim = SGD(lr=1)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

optim = SGD(lr=10)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 1, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=1000)

optim = SGD(lr=0.1)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])

Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 10, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=1000)

optim = SGD(lr=0.1)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])

Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=1000)

optim = SGD(lr=0.1)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])

Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 100, kernel_initializer= pesos_inicial, input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=1000)

optim = SGD(lr=0.1)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])

Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial , input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

optim = Adam(lr=0.1)
MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(20,20))
for i,nice in enumerate(Correctos[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(x_test[nice].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
              "Predict:{}, Real:{}".format(resultado[nice],
                                            y_test[nice]))
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(20,20))
for i,nice in enumerate(Incorrectos[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(x_test[nice].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
              "Predict:{}, Real:{}".format(resultado[nice],
                                            y_test[nice]))
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
from keras.regularizers import l2

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, kernel_regularizer=l2() , input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=15)

MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")


Epocas = 100

MLP = Sequential()
#Distribucion para inicializar los pesos
pesos_inicial = initializers.RandomUniform(minval=-0.2,maxval=0.2)
#capa oculta
MLP.add(Dense(units= 25, kernel_initializer= pesos_inicial, kernel_regularizer=l2() , input_dim= 784, activation= activations.sigmoid))
#output
MLP.add(Dense(2, activation = activations.softmax))

earlyStopping = EarlyStopping(monitor='val_loss',patience=1000)

MLP.compile(loss='categorical_crossentropy', optimizer = optim, metrics=['accuracy'])
Historia = MLP.fit( x_train, Y_train, 
                    epochs=Epocas,
                    batch_size=32, #mini batch
                    callbacks=[earlyStopping], #Ojala funcione
                    validation_data = (x_val, Y_val))
resultado = MLP.predict_classes(x_test)
Correctos = np.nonzero(resultado == y_test)[0]
Incorrectos = np.nonzero(resultado != y_test)[0]

print("Set de entrenamiento: ", y_test.shape[0])
print("Acertados: ",len(Correctos))
print("No acertados: ",len(Incorrectos))
plt.figure(figsize=(12,10))
plt.plot(Historia.history['loss'], label='training data')
plt.plot(Historia.history['val_loss'], label='validation data')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")
plt.show()
plt.figure(figsize=(12,10))
plt.plot(Historia.history['accuracy'], label='accuracy')
plt.plot(Historia.history['val_accuracy'], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")