import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy as cp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
trainPure    = np.load("../input/train_images_pure.npy")
trainRotated = np.load("../input/train_images_rotated.npy")
trainNoisy   = np.load("../input/train_images_noisy.npy")
trainBoth    = np.load("../input/train_images_both.npy")
testImages   = np.load("../input/Test_images.npy")
trainLabels = pd.read_csv("../input/train_labels.csv", header=0, index_col=0, na_values="?")
trainLabels.head()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(trainPure[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(trainPure[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(trainPure[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(trainPure[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(trainRotated[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(trainRotated[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(trainRotated[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(trainRotated[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(trainNoisy[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(trainNoisy[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(trainNoisy[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(trainNoisy[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(trainBoth[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(trainBoth[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(trainBoth[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(trainBoth[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
# Mantendo a seed constante para poder reproduzir os resultados
np.random.seed(7)
trainPure    = trainPure.reshape(trainPure.shape[0],       1, 28, 28).astype('float32')
trainRotated = trainRotated.reshape(trainRotated.shape[0], 1, 28, 28).astype('float32')
trainNoisy   = trainNoisy.reshape(trainNoisy.shape[0],     1, 28, 28).astype('float32')
trainBoth    = trainBoth.reshape(trainBoth.shape[0],       1, 28, 28).astype('float32')
testImages   = testImages.reshape(testImages.shape[0],     1, 28, 28).astype('float32')
# Normalizando as features
trainPure    = trainPure    / 255
trainRotated = trainRotated / 255
trainNoisy   = trainNoisy   / 255
trainBoth    = trainBoth    / 255
testImages   = testImages   / 255
# one hot encode outputs
trainLabels = np_utils.to_categorical(trainLabels)
nClasses = trainLabels.shape[1]
def baseline_model():
    # criando o modelo
    model = Sequential()
    # de novo, simplesmente basta adicionar a camada de convolução. Vale notar que para esta camada temos que determinar
    # o número de kernels de convolução, (32,neste caso), o tamanho do kernel de convolução (5x5) e o formato de cada 
    # entrada, neste caso, 1 canal e uma imagem 28x28
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # incluimos uma camada de dropout para adicionar robustez ao aprendizado
    model.add(Dropout(0.2))
    # em seguida, para a camada de saída, temos que formatar as saídas da camada convolucional em um vetor que pode servir
    # de entrada para uma camada conectada tradicional
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # adicionamos, por fim, a camada de saída, com ativação softmax para transformar os valores em probabilidades.
    model.add(Dense(nClasses, activation='softmax'))
    # compilando o modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
model.summary()
trainPure,    xValidPure,    labelsPure,    yValidPure    = train_test_split(trainPure,    trainLabels, test_size = 0.2)
trainRotated, xValidRotated, labelsRotated, yValidRotated = train_test_split(trainRotated, trainLabels, test_size = 0.2)
trainNoisy,   xValidNoisy,   labelsNoisy,   yValidNoisy   = train_test_split(trainNoisy,   trainLabels, test_size = 0.2)
trainBoth,    xValidBoth,    labelsBoth,    yValidBoth    = train_test_split(trainBoth,    trainLabels, test_size = 0.2)
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fitando o modelo PURE
model.fit(trainPure, labelsPure, validation_data=(xValidPure, yValidPure), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidPure, yValidPure, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Fitando o modelo ROTATED
model.fit(trainRotated, labelsRotated, validation_data=(xValidRotated, yValidRotated), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidRotated, yValidRotated, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Fitando o modelo NOISY
model.fit(trainNoisy, labelsNoisy, validation_data=(xValidNoisy, yValidNoisy), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidNoisy, yValidNoisy, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Fitando o modelo BOTH
model.fit(trainBoth, labelsBoth, validation_data=(xValidBoth, yValidBoth), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidBoth, yValidBoth, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
probabilities = model.predict(testImages)
predictedData = np.argmax(probabilities, axis=1)
predictedData
aux = pd.DataFrame(testImages[:,0,0,0])
output = pd.DataFrame(aux.index)
output["label"] = predictedData
output.columns = ["Id", "label"]
output.head()
output.to_csv("PMR3508_MarcusPavani_FashionMNIST_NoPooling.csv", index=False)
def baseline_model():
    # criando o modelo
    model = Sequential()
    # de novo, simplesmente basta adicionar a camada de convolução. Vale notar que para esta camada temos que determinar
    # o número de kernels de convolução, (32,neste caso), o tamanho do kernel de convolução (5x5) e o formato de cada 
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
    model.add(Dense(nClasses, activation='softmax'))
    # compilando o modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
model.summary()
# Fitando o modelo PURE
model.fit(trainPure, labelsPure, validation_data=(xValidPure, yValidPure), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidPure, yValidPure, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Fitando o modelo ROTATED
model.fit(trainRotated, labelsRotated, validation_data=(xValidRotated, yValidRotated), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidRotated, yValidRotated, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Fitando o modelo NOISY
model.fit(trainNoisy, labelsNoisy, validation_data=(xValidNoisy, yValidNoisy), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidNoisy, yValidNoisy, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Fitando o modelo BOTH
model.fit(trainBoth, labelsBoth, validation_data=(xValidBoth, yValidBoth), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
# Avaliação final do modelo
scores = model.evaluate(xValidBoth, yValidBoth, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
probabilities = model.predict(testImages)
predictedData = np.argmax(probabilities, axis=1)
predictedData
aux = pd.DataFrame(testImages[:,0,0,0])
output = pd.DataFrame(aux.index)
output["label"] = predictedData
output.columns = ["Id", "label"]
output.head()
output.to_csv("PMR3508_MarcusPavani_FashionMNIST_WithPooling.csv", index=False)