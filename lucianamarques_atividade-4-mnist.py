import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

# Carregamento das bases de treino
images_pure = np.load("../input/train_images_pure.npy")
images_rotated = np.load("../input/train_images_rotated.npy")
images_noisy = np.load("../input/train_images_noisy.npy")
images_both = np.load("../input/train_images_both.npy")

# Labels de treino

train_labels = pd.read_csv("../input/train_labels.csv")

train_labels.drop(columns = "Id")

# Base de teste

test_images = np.load("../input/Test_images.npy")

#plt.imshow(test_images[len(test_images) - 1])
import matplotlib.pyplot as plt

# análise das bases

# n é o número de imagens - 60000
n = len(images_pure)

# tamanho da primeira imagem - array de 28 posições
ki = len(images_pure[0])

# tamanho do primeiro termo da primeira imagem - array de 28 posições
kl = len(images_pure[0][0])

print(n, ki, kl)



# Base pura
plt.subplot(2,2,1)
plt.imshow(images_pure[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(images_pure[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(images_pure[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(images_pure[3], cmap=plt.get_cmap('gray'))
plt.show()

# Base com as figuras rotacionadas
plt.subplot(2,2,1)
plt.imshow(images_rotated[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(images_rotated[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(images_rotated[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(images_rotated[3], cmap=plt.get_cmap('gray'))
plt.show()
# Base com ruído
plt.subplot(2,2,1)
plt.imshow(images_noisy[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(images_noisy[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(images_noisy[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(images_noisy[3], cmap=plt.get_cmap('gray'))
plt.show()
# Base com ruído e rotacionada
plt.subplot(2,2,1)
plt.imshow(images_both[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(images_both[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(images_both[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(images_both[3], cmap=plt.get_cmap('gray'))
plt.show()
images_test = np.load('../input/Test_images.npy')

#print(len(images_test[0]))

plt.subplot(2,2,1)
plt.imshow(images_test[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(images_test[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(images_test[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(images_test[3], cmap=plt.get_cmap('gray'))
plt.show()
# Importando Keras

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

seed = 7
np.random.seed(seed)

#labels de treino
y_train = pd.read_csv("../input/train_labels.csv",index_col=0)

num_pixels = images_pure.shape[1] * images_pure.shape[2]

Xpure = images_pure.reshape(images_pure.shape[0], num_pixels).astype('float32')
Xnoisy = images_noisy.reshape(images_noisy.shape[0], num_pixels).astype('float32')
Xrotated = images_rotated.reshape(images_rotated.shape[0], num_pixels).astype('float32')
Xboth = images_both.reshape(images_both.shape[0], num_pixels).astype('float32')

Xpure = Xpure / 255
Xnoisy = Xnoisy / 255
Xrotated = Xrotated / 255
Xboth = Xboth / 255

Xtest = images_test.reshape(images_test.shape[0], num_pixels).astype('float32')
#Separação da base de treino em pedaços para validação

from sklearn.model_selection import train_test_split

Xpure_train, x_val, Y_train, y_val = train_test_split(Xpure, y_train, test_size=0.2)

#Callback 

from keras.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]

Y_train = np_utils.to_categorical(Y_train)
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
num_classes = y_val.shape[1]
#definição do modelo 
def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#testando modelo
%matplotlib inline
# construimos o modelo com nossa classe
model = baseline_model()
model.summary()

# Fit do modelo

model.fit(Xpure_train, Y_train, validation_data=(x_val, y_val), epochs=10, batch_size=200, verbose=2, callbacks = callbacks)

# Avaliação final do modelo.
scores = model.evaluate(Xpure, y_train, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

scores = model.evaluate(Xnoisy, y_train, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

scores = model.evaluate(Xrotated, y_train, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

scores = model.evaluate(Xboth, y_train, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

def baseline_model():
    # criando o modelo
    model = Sequential()
    # de novo, simplesmente basta adicionar a camada de convolução. Vale notar que para esta camada temos que determinar
    #o número de kernels de convolução, (32,neste caso), o tamanho do kernel de convolução (5x5) e o formato de cada 
    # entrada, neste caso, 1 canal e uma imagem 28x28
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # em seguida, adicionamos uma camada de pooling, para reduzir a dimensionalidade do problema e acelearar o aprendizado
    # sobre o efeito do pooling, vc pode ler mais aqui: https://pt.coursera.org/lecture/convolutional-neural-networks/pooling-layers-hELHk
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # incluimos uma camada de dropout para adicionar robustez ao aprendizado
    model.add(Dropout(0.2))
    # em seguida, para a camada de saída, temos que formatar as saídas da camada convolucional em um vetor que pode servir
    # de entrada para uma camada conectada tradicional
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # adicionamos, por fim, a camada de saída, com ativação softmax para transformar os valores em probabilidades.
    model.add(Dense(num_classes, activation='softmax'))
    # compilando o modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_pool = baseline_model()
model_pool.summary()
model.fit(Xpure_train, Y_train, validation_data=(x_val, y_val), epochs=10, batch_size=200, verbose=2, callbacks = callbacks)
