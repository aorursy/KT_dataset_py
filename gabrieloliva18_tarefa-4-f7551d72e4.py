import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('th')
puro = np.load('../input/imagesdb/train_images_pure.npy')
plt.subplot(331)
plt.imshow(puro[0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(puro[1], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(puro[2], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(puro[3], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(puro[4], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(puro[5], cmap=plt.get_cmap('gray'))
ruido = np.load('../input/imagesdb/train_images_noisy.npy')
plt.subplot(331)
plt.imshow(ruido[0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(ruido[1], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(ruido[2], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(ruido[3], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(ruido[4], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(ruido[5], cmap=plt.get_cmap('gray'))
rot = np.load('../input/imagesdb/train_images_rotated.npy')
plt.subplot(331)
plt.imshow(rot[0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(rot[1], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(rot[2], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(rot[3], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(rot[4], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(rot[5], cmap=plt.get_cmap('gray'))
misto = np.load('../input/imagesdb/train_images_both.npy')
plt.subplot(331)
plt.imshow(misto[0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(misto[1], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(misto[2], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(misto[3], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(misto[4], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(misto[5], cmap=plt.get_cmap('gray'))
#Seed constante

seed = 7
np.random.seed(seed)
#Transformando imagens em vetor

Ytrain = pd.read_csv("../input/imagesdb/train_labels.csv",index_col=0)
Xpuro = puro.reshape(puro.shape[0], 1, 28, 28).astype('float32')
Xruido = ruido.reshape(ruido.shape[0], 1, 28, 28).astype('float32')
Xrot = rot.reshape(rot.shape[0], 1, 28, 28).astype('float32')
Xmisto = misto.reshape(misto.shape[0], 1, 28, 28).astype('float32')
#Normalização de inputs

Xpuro = Xpuro / 255
Xruido = Xruido / 255
Xrot = Xrot / 255
Xmisto = Xmisto / 255

#One-hot encode outputs

Ytrain = np_utils.to_categorical(Ytrain)
num_classes = Ytrain.shape[1]
#Divisão do banco para treino e teste

Xtrain,Xtest,Ytrain2,Ytest = train_test_split(Xpuro,Ytrain, test_size = 0.25)
#Criacao do modelo da rede

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
modelo = baseline_model()
modelo.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
modelo.fit(Xtrain, Ytrain2, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 1500, verbose=1, callbacks = callbacks)
#Validacao banco Puro (para validacao)

scores = modelo.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco Rotacionado

scores = modelo.evaluate(Xrot, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco misto

scores = modelo.evaluate(Xmisto, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco ruido

scores = modelo.evaluate(Xruido, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco puro

scores = modelo.evaluate(Xpuro, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
Xtrain,Xtest,Ytrain2,Ytest = train_test_split(Xmisto,Ytrain, test_size = 0.25)
modelo2 = baseline_model()
modelo2.fit(Xtrain, Ytrain2, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 1500, verbose=1, callbacks = callbacks)
#Validacao banco misto (base de treino para validacao)

scores = modelo2.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco rotacionado

scores = modelo2.evaluate(Xrot, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco misto

scores = modelo2.evaluate(Xmisto, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco puro

scores = modelo2.evaluate(Xpuro, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco ruido

scores = modelo2.evaluate(Xruido, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
def modelopooling():
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
modelo3 = modelopooling()
modelo3.summary()
Xtrain,Xtest,Ytrain2,Ytest = train_test_split(Xpuro,Ytrain, test_size = 0.25)
modelo3.fit(Xtrain, Ytrain2, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 1000, verbose=1, callbacks = callbacks)
#Validacao banco Puro (para validacao)

scores = modelo3.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco rotacionado

scores = modelo3.evaluate(Xrot, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco misto

scores = modelo3.evaluate(Xmisto, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco ruido

scores = modelo3.evaluate(Xruido, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo4 = modelopooling()
Xtrain,Xtest,Ytrain2,Ytest = train_test_split(Xmisto,Ytrain, test_size = 0.25)
modelo4.fit(Xtrain, Ytrain2, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 700, verbose=1, callbacks = callbacks)
#Validacao banco misto (parte de validacao)

scores = modelo4.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco rotacionado

scores = modelo4.evaluate(Xrot, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco misto

scores = modelo4.evaluate(Xmisto, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Validacao banco ruido

scores = modelo4.evaluate(Xruido, Ytrain, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
Teste = np.load('../input/imagesdb/Test_images.npy')
Teste = Teste.reshape(Teste.shape[0], 1, 28, 28).astype('float32')
Xteste = Teste / 255
Pred = modelo4.predict_classes(Xteste)
result = pd.DataFrame(columns = ['Id','label'])
result.label = Pred
result.Id = range(len(Teste))
result.to_csv("result.csv",index=False)
