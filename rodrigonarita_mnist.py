from IPython.display import Image
from IPython.core.display import HTML 
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
import random
import imutils
from sklearn.model_selection import train_test_split
import operator

import sys
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import Iterator

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
train_labels = pd.read_csv("/Users/Rodrigo Narita/Documents/Python Scripts/MNIST/train_labels.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train_labels.head()
# carregamos o dataset:
train_pure=np.load("/Users/Rodrigo Narita/Documents/Python Scripts/MNIST/train_images_pure.npy")
plt.subplot(221)
plt.imshow(train_pure[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(train_pure[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(train_pure[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(train_pure[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
random.seed(10)
rotations=[]
for i in range(60000):
   rnum=random.randint(-45, 45)
   rotations.append(rnum)
rotations=np.array(rotations)
i=0
randomly_rotated=[]
for element in train_pure:
    aux=imutils.rotate(element, rotations[i])
    i=i+1
    randomly_rotated.append(aux)
randomly_rotated=np.array(randomly_rotated)
plt.subplot(221)
plt.imshow(randomly_rotated[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
random.seed(10)
train_data0,test_data0,rotations_train,validation = train_test_split(randomly_rotated,rotations, test_size = 0.2)
# transformando as imagens 28x28 em um vetor com 784 componentes para poder servir de entrada para a primeira camada da rede
num_pixels = train_data0[1].size
train_data1 = train_data0.reshape(train_data0.shape[0], num_pixels).astype('float32')
test_data1 = test_data0.reshape(test_data0.shape[0], num_pixels).astype('float32')
# Normalizando os inputs - prática comum em NN - deixa o problema melhor comportado numericamente
train_data1 = train_data1 / 255
test_data1 = test_data1 / 255
labels1=np.ravel(rotations_train)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels1)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded1)
num_classes = onehot_encoded1.shape[1]
labels2=np.ravel(validation)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels2)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded2)
num_classes = onehot_encoded2.shape[1]
# definindo o modelo baseline
def rotation_model():
    # criamos o modelo
    # começamos por instanciar o esqueleto do modelo, neste caso, o sequencial
    model = Sequential()
    # em seguida, adicionamos as camadas uma a uma. Adicionamos uma camada de entrada com ativação relu
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # e a camada de saída, com ativação softmax para transformar as saídas numéricas em probabilidades
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compilamos o modelo escolhendo a função objetiva, o otimizador (cuja escolha é empírica) e a métrica 
    #de performance mais conveniente.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
%matplotlib inline
# construimos o modelo com nossa classe
model1 = rotation_model()
model1.summary()
model1.fit(train_data1, onehot_encoded1, validation_data=(test_data1, onehot_encoded2), epochs=10, batch_size=200, verbose=2)
# Avaliação final do modelo.
scores = model1.evaluate(test_data1, onehot_encoded2, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
scores
results=model1.predict(test_data1)
i=0
rotations=[]
for element in results:
    index, value = max(enumerate(element), key=operator.itemgetter(1))
    i=i+1
    rotations.append(45-index)
rotations
depois=imutils.rotate(test_data0[5], rotations[5])
plt.subplot(221)
plt.imshow(test_data0[5], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem rotacionada')
plt.subplot(222)
plt.imshow(depois, interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem tratada')
plt.show()
# carregamos o dataset:
train_noisy=np.load("/Users/Rodrigo Narita/Documents/Python Scripts/MNIST/train_images_noisy.npy")
plt.subplot(221)
plt.imshow(train_noisy[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(train_noisy[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(train_noisy[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(train_noisy[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
kernel1 = np.ones((1,2), np.uint8) 
eroded_image = cv2.erode(train_noisy, kernel1, iterations=1) 
plt.subplot(221)
plt.imshow(eroded_image[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(eroded_image[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(eroded_image[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(eroded_image[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
kernel2 = np.ones((1,3), np.uint8) 
filtered_image = cv2.dilate(eroded_image, kernel2, iterations=1)
plt.subplot(221)
plt.imshow(filtered_image[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(filtered_image[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(filtered_image[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(filtered_image[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
# carregamos o dataset:
train_rotated=np.load("/Users/Rodrigo Narita/Documents/Python Scripts/MNIST/train_images_rotated.npy")
plt.subplot(221)
plt.imshow(train_rotated[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(train_rotated[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(train_rotated[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(train_rotated[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
# carregamos o dataset:
train_both=np.load("/Users/Rodrigo Narita/Documents/Python Scripts/MNIST/train_images_both.npy")
plt.subplot(221)
plt.imshow(train_both[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(train_both[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(train_both[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(train_both[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
train_images_vector = train_both.reshape(train_both.shape[0], num_pixels).astype('float32')
train_images_vector = train_images_vector / 255
labels=train_labels.drop(['Id'], axis=1)
labels=np.ravel(labels)
results1=model1.predict(train_images_vector)
i=0
rotations1=[]
for element in results1:
    index, value = max(enumerate(element), key=operator.itemgetter(1))
    i=i+1
    rotations1.append(45-index)
i=0
final_images=[]
for element1 in train_both:
    final_images.append(imutils.rotate(element1, rotations1[i]))
    i=i+1
final_images=np.array(final_images)
i=0
eroded_images=[]
for element in final_images:
    eroded_images.append(cv2.erode(element, kernel1, iterations=1) )
    i=i+1
i=0
filtered_images=[]
for element in eroded_images:
    filtered_images.append(cv2.dilate(element, kernel2, iterations=1) )
    i=i+1
filtered_images=np.array(filtered_images)
random.seed(10)
train_data_both,test_data_both,train_labels_0,validation1 = train_test_split(filtered_images,labels, test_size = 0.2)
# transformando as imagens 28x28 em um vetor com 784 componentes para poder servir de entrada para a primeira camada da rede
num_pixels = train_data_both[1].size
train_data2 = train_data_both.reshape(train_data_both.shape[0], num_pixels).astype('float32')
test_data2 = test_data_both.reshape(test_data_both.shape[0], num_pixels).astype('float32')
# Normalizando os inputs - prática comum em NN - deixa o problema melhor comportado numericamente
train_data2 = train_data2 / 255
test_data2 = test_data2 / 255
train_labels_0=train_labels_0.drop(['Id'], axis=1)
train_labels_0=np.ravel(train_labels_0)
validation1=validation1.drop(['Id'], axis=1)
validation1=np.ravel(validation1)
labels_tr=np.ravel(train_labels_0)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels_tr)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_tr = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded_tr)
num_classes = onehot_encoded_tr.shape[1]
labels_te=np.ravel(validation1)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels_te)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_te = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded_te)
# definindo o modelo baseline
def final_model():
    # criamos o modelo
    # começamos por instanciar o esqueleto do modelo, neste caso, o sequencial
    model = Sequential()
    # em seguida, adicionamos as camadas uma a uma. Adicionamos uma camada de entrada com ativação relu
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # e a camada de saída, com ativação softmax para transformar as saídas numéricas em probabilidades
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compilamos o modelo escolhendo a função objetiva, o otimizador (cuja escolha é empírica) e a métrica 
    #de performance mais conveniente.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
%matplotlib inline
# construimos o modelo com nossa classe
model_last = final_model()
model_last.summary()
model_last.fit(train_data2, onehot_encoded_tr, validation_data=(test_data2, onehot_encoded_te), epochs=10, batch_size=200, verbose=2)
# Avaliação final do modelo.
scores = model_last.evaluate(test_data2, onehot_encoded_te, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# carregamos o dataset:
test_images=np.load("/Users/Rodrigo Narita/Documents/Python Scripts/MNIST/Test_images.npy")
plt.subplot(221)
plt.imshow(test_images[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(test_images[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(test_images[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(test_images[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
eroded_test = cv2.erode(test_images, kernel1, iterations=1) 
filtered_test = cv2.dilate(eroded_test, kernel2, iterations=1)
train_data=train_pure
test_data=train_both
# transformando as imagens 28x28 em um vetor com 784 componentes para poder servir de entrada para a primeira camada da rede
num_pixels = train_data[1].size
train_data = train_data.reshape(train_data.shape[0], num_pixels).astype('float32')
test_data = test_data.reshape(test_data.shape[0], num_pixels).astype('float32')
# Normalizando os inputs - prática comum em NN - deixa o problema melhor comportado numericamente
train_data = train_data / 255
test_data = test_data / 255
labels=train_labels.drop(['Id'], axis=1)
labels=np.ravel(labels)
train_labels.shape
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
num_classes = onehot_encoded.shape[1]
# definindo o modelo baseline
def baseline_model():
    # criamos o modelo
    # começamos por instanciar o esqueleto do modelo, neste caso, o sequencial
    model = Sequential()
    # em seguida, adicionamos as camadas uma a uma. Adicionamos uma camada de entrada com ativação relu
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # e a camada de saída, com ativação softmax para transformar as saídas numéricas em probabilidades
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compilamos o modelo escolhendo a função objetiva, o otimizador (cuja escolha é empírica) e a métrica 
    #de performance mais conveniente.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
%matplotlib inline
# construimos o modelo com nossa classe
model = baseline_model()
model.summary()
model.fit(train_data, onehot_encoded, validation_data=(test_data, onehot_encoded), epochs=10, batch_size=200, verbose=2)
# Avaliação final do modelo.
scores = model.evaluate(test_data, onehot_encoded, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
test_images_vector = test_images.reshape(test_images.shape[0], num_pixels).astype('float32')
test_images_vector = test_images_vector / 255
results1=model1.predict(test_images_vector)
i=0
rotations1=[]
for element in results1:
    index, value = max(enumerate(element), key=operator.itemgetter(1))
    i=i+1
    rotations1.append(45-index)
depois1=imutils.rotate(test_images[3], rotations1[3])
plt.subplot(221)
plt.imshow(test_images[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem rotacionada')
plt.subplot(222)
plt.imshow(depois1, interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem tratada')
plt.show()
kernel1 = np.ones((1,2), np.uint8) 
eroded_image1 = cv2.erode(depois1, kernel1, iterations=1) 
kernel2 = np.ones((1,3), np.uint8) 
filtered_image1 = cv2.dilate(eroded_image1, kernel2, iterations=1)
plt.subplot(221)
plt.imshow(test_images[3], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem de teste')
plt.subplot(222)
plt.imshow(depois1, interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem rotacionada')
plt.subplot(223)
plt.imshow(filtered_image1, interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('Imagem filtrada')
plt.show()
i=0
final_images=[]
for element1 in test_images:
    final_images.append(imutils.rotate(element1, rotations1[i]))
    i=i+1
plt.subplot(221)
plt.imshow(final_images[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('0')
plt.subplot(222)
plt.imshow(final_images[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('1')
plt.subplot(223)
plt.imshow(final_images[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('2')
plt.show()
final_images=np.array(final_images)
i=0
eroded_images=[]
for element in final_images:
    eroded_images.append(cv2.erode(element, kernel1, iterations=1) )
    i=i+1
i=0
filtered_images=[]
for element in eroded_images:
    filtered_images.append(cv2.dilate(element, kernel2, iterations=1) )
    i=i+1
plt.subplot(221)
plt.imshow(filtered_images[0], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('0')
plt.subplot(222)
plt.imshow(filtered_images[1], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('1')
plt.subplot(223)
plt.imshow(filtered_images[2], interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.title('2')
plt.show()
filtered_images=np.array(filtered_images)
filtered = filtered_images.reshape(filtered_images.shape[0], num_pixels).astype('float32')
filtered = filtered / 255
results_final=model_last.predict(filtered)
i=0
resultados=[]
for element in results_final:
    index, value = max(enumerate(element), key=operator.itemgetter(1))
    i=i+1
    resultados.append(index)
resultados
resultados=np.array(resultados)
np.savetxt("resultados.csv", resultados, delimiter=",")