import numpy as np # Vamos guardar nossos dados como vetor de numpy
import os # Para trabalhar com diretórios
from PIL import Image # Para manipular as imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotagem


lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('../input/novoleapgestrecog/leapGestRecog/00'):
    if not j.startswith('.'): # Para garantir que não leia pastas ocultas
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup
x_data = []
y_data = []
datacount = 0 # Contador de quantas imagens há no dataset
for i in range(0, 5): # iterar sob as 10 pastas maior nível
    for j in os.listdir('../input/novoleapgestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Evitando pastas escondidas
            count = 0  # Para contar imagens de um gesto
            for k in os.listdir('../input/novoleapgestrecog/leapGestRecog/0' + 
                                str(i) + '/' + j + '/'):
                                # Iterar nas imagens
                img = Image.open('../input/novoleapgestrecog/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Le e converte para escala de cinza
                img = img.resize((200, 200))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Redimensiona para o tamanho correto
images = x_data
print("Número de imagens:" + str(datacount))
from random import randint
for i in range(0, 9):
    plt.imshow(x_data[i*300 , :, :])
    plt.title(reverselookup[y_data[i*300 ,0]])
    plt.show()

import keras
from keras.utils import to_categorical
y_data = to_categorical(y_data)
print(y_data)
x_data = x_data.reshape((datacount, 200, 200, 1))
x_data /= 255
print(x_data)
from sklearn.model_selection import train_test_split
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)
from keras import layers
from keras import models
classificador = models.Sequential()
classificador.add(layers.Conv2D(32, (3,3), strides=(2, 2), input_shape = (200, 200, 1), activation = 'relu'))
classificador.add(layers.BatchNormalization())
classificador.add(layers.MaxPooling2D(pool_size = (2,2)))

classificador.add(layers.Conv2D(64, (3,3), activation = 'relu'))
classificador.add(layers.BatchNormalization())
classificador.add(layers.MaxPooling2D(pool_size = (2,2)))

classificador.add(layers.Conv2D(64, (3,3), activation = 'relu'))
classificador.add(layers.BatchNormalization())
classificador.add(layers.MaxPooling2D(pool_size = (2,2)))

classificador.add(layers.Flatten())

classificador.add(layers.Dense(units = 128, activation = 'relu'))
classificador.add(layers.Dense(10, activation='softmax'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
classificador.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_validate, y_validate))
import sklearn.metrics as metrics
from sklearn.metrics import classification_report

y_predict = classificador.predict(x_test)
y_pred_labels = np.argmax(y_predict, axis=1)

y_true = np.argmax(y_test, axis=1)

confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_labels)
print(confusion_matrix)
print('\n\n')
print(classification_report(y_true, y_pred_labels, 
 target_names=[reverselookup[0], reverselookup[1],reverselookup[2], reverselookup[3], reverselookup[4], reverselookup[5],reverselookup[6],reverselookup[7],reverselookup[8],reverselookup[9]]))
[erro, acuracia] = classificador.evaluate(x_test,y_test,verbose=1)
print("Acurácia:" + str(acuracia))