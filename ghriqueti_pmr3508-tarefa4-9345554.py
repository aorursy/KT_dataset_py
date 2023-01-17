from IPython.display import Image
from IPython.core.display import HTML
!python -m pip install --upgrade pip
# se vc ainda não tem as bibliotecas, execute essa linha:
!pip install --user theano 
!pip install --user keras 
!pip install --user tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.utils import np_utils
from keras import backend as K
from keras import initializers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('th')
# fixando a random seed:
seed = 9
np.random.seed(seed)
# carregando dados
trainY = pd.read_csv('../input/9345554data/train_labels.csv', names=["Id","label"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)
trainXPure = np.load("../input/9345554data/train_images_pure.npy")
trainXRot = np.load("../input/9345554data/train_images_rotated.npy")
trainXNoisy = np.load("../input/9345554data/train_images_noisy.npy")
trainXBoth = np.load("../input/9345554data/train_images_both.npy")
testX = np.load("../input/9345554data/Test_images.npy")
sample = pd.read_csv("../input/9345554data/sample_sub.csv", names=["Id","label"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)
import matplotlib.pyplot as plt
%matplotlib inline
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(trainXPure[2], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(trainXRot[2], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(trainXNoisy[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(trainXBoth[2], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
num_pixels = trainXPure.shape[1]*trainXPure.shape[2]
num_side = trainXPure.shape[1]
# reshape to be [samples][pixels][width][height]
trainXPureCNN = trainXPure.reshape(trainXPure.shape[0], 1, 28, 28).astype('float32')
trainXRotCNN = trainXRot.reshape(trainXRot.shape[0], 1, 28, 28).astype('float32')
trainXNoisyCNN = trainXNoisy.reshape(trainXNoisy.shape[0], 1, 28, 28).astype('float32')
trainXBothCNN = trainXBoth.reshape(trainXBoth.shape[0], 1, 28, 28).astype('float32')
testXCNN = testX.reshape(testX.shape[0], 1, 28, 28).astype('float32')
# normalizando os inputs 0-255 to 0-1
trainXPureCNN = trainXPureCNN / 255
trainXRotCNN = trainXRotCNN / 255
trainXNoisyCNN = trainXNoisyCNN / 255
trainXBothCNN = trainXBothCNN / 255
testXCNN = testXCNN / 255
# carregando dados
trainY = pd.read_csv('../input/9345554data/train_labels.csv', names=["Id","label"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)
trainXPure = np.load("../input/9345554data/train_images_pure.npy")
trainXRot = np.load("../input/9345554data/train_images_rotated.npy")
trainXNoisy = np.load("../input/9345554data/train_images_noisy.npy")
trainXBoth = np.load("../input/9345554data/train_images_both.npy")
testX = np.load("../input/9345554data/Test_images.npy")
sample = pd.read_csv("../input/9345554data/sample_sub.csv", names=["Id","label"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)
# transformando as imagens 28x28 em um vetor com 784 componentes para poder servir de entrada para a primeira camada da rede
trainXPure = trainXPure.reshape(trainXPure.shape[0], num_pixels).astype('float32')
trainXRot = trainXRot.reshape(trainXRot.shape[0], num_pixels).astype('float32')
trainXNoisy = trainXNoisy.reshape(trainXNoisy.shape[0], num_pixels).astype('float32')
trainXBoth = trainXBoth.reshape(trainXBoth.shape[0], num_pixels).astype('float32')
testX = testX.reshape(testX.shape[0], num_pixels).astype('float32')
# normalizando os inputs 0-255 to 0-1
trainXPure = trainXPure / 255
trainXRot = trainXRot / 255
trainXNoisy = trainXNoisy / 255
trainXBoth = trainXBoth / 255
testX = testX / 255
# one hot encode outputs
trainYl = trainY["label"]
trainYl = np_utils.to_categorical(trainYl)
num_classes = trainYl.shape[1]
X_train,X_val,Y_trainB,Y_val = train_test_split(trainXBothCNN, trainYl, test_size = 0.2)
def mao_digito_model():
    # criando modelo
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build model
model = mao_digito_model()
model.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, 
          batch_size=200, verbose=1, callbacks = callbacks)
# Final evaluation of the model
scores = model.evaluate(trainXBothCNN, trainYl, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Final evaluation of the model
scores = model.evaluate(trainXRotCNN, trainYl, verbose=0)
print("Rot Error: %.2f%%" % (100-scores[1]*100))
# Final evaluation of the model
scores = model.evaluate(trainXNoisyCNN, trainYl, verbose=0)
print("Noisy Error: %.2f%%" % (100-scores[1]*100))
# Final evaluation of the model
scores = model.evaluate(trainXPureCNN, trainYl, verbose=0)
print("Pure Error: %.2f%%" % (100-scores[1]*100))
def igual_model():
    model = Sequential()
    model.add(Conv2D(1, (5, 5), padding='same', input_shape=(1, 28, 28), activation='relu'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
%matplotlib inline
# construimos o modelo com nossa classe
modeli = igual_model()
modeli.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
modeli.fit(trainXPureCNN, trainXPureCNN, validation_data=(trainXNoisyCNN, trainXNoisyCNN), epochs=10, 
          batch_size=400, verbose=1, callbacks = callbacks)
# Final evaluation of the model
scores = modeli.evaluate(trainXBothCNN, trainXBothCNN, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
def ruido_model():
    model = Sequential()
    model.add(Conv2D(1, (5, 5), padding='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))
#    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_side, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
%matplotlib inline
# construimos o modelo com nossa classe
modelru = ruido_model()
modelru.summary()
Noisy_train, Noisy_val, Pure_train, Pure_val = train_test_split(trainXNoisyCNN, trainXPureCNN, test_size = 0.2)
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
modelru.fit(Noisy_train, Pure_train, validation_data=(Noisy_val, Pure_val), epochs=20, 
          batch_size=40, verbose=1, callbacks = callbacks)
# Final evaluation of the model
scores = modelru.evaluate(trainXBothCNN, trainXRotCNN, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
def rot_model():
    model = Sequential()
    model.add(Conv2D(1, (5, 5), padding='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))
#    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_side, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build model
modelrot = rot_model()
modelrot.summary()
Both_train, Both_val, Noisy_train, Noisy_val = train_test_split(trainXBothCNN, trainXNoisyCNN, test_size = 0.2)
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
modelrot.fit(Both_train, Noisy_train, validation_data=(Both_val, Noisy_val), epochs=20, 
          batch_size=400, verbose=1, callbacks = callbacks)
# Final evaluation of the model
scores = modelrot.evaluate(trainXRotCNN, trainXPureCNN, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
X_train,X_val,Y_train,Y_val = train_test_split(trainXPureCNN, trainYl, test_size = 0.2)
def mao_digito_model():
    # criando modelo
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build model
model = mao_digito_model()
model.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, 
          batch_size=200, verbose=1, callbacks = callbacks)
# Final evaluation of the model
scores = model.evaluate(trainXBothCNN, trainYl, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
testYp = model.predict(testXCNN)
testYpred = np.argmax(testYp, axis=1)
testYpred
aux = pd.DataFrame(testXCNN[:,0,0,0])
testY = pd.DataFrame(aux.index)
testY["label"] = testYpred
testY.columns = ["Id", "label"]
testY.head()
testY.to_csv("PMR3508_9345554_submition.csv", index=False)