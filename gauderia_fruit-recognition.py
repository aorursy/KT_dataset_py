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
import keras as ks

from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

from keras.models import Sequential

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline
#imagens de teste

imagens = []

label = []

shape = (64,64)



for i in os.listdir('../input/test_zip/test'):

        img = cv2.imread(os.path.join('../input/test_zip/test',i))

        if i.split('.')[1] == 'jpg':

            img2 = cv2.resize(img,shape)

            imagens.append(img2)

            label.append(i.split('_')[0])
#imagens de treino

imagens_test = []

labels_test = []



for i in os.listdir('../input/train_zip/train'):

        img = cv2.imread(os.path.join('../input/train_zip/train',i))

        if i.split('.')[1] =='jpg':

            img2 = cv2.resize(img,shape)

            imagens_test.append(img2)

            labels_test.append(i.split('_')[0])



output_test = pd.get_dummies(labels_test)
#printando imagem

plt.imshow(imagens[15])

plt.title(label[15])
#verificando o shape das imagens

imagens[0].shape
#OneHot nas labels(output)

labels = pd.get_dummies(label).values
#Transformando as imagens de input em arrays

imagens = np.array(imagens)
#input e output

input_shape = imagens[0].shape

output = len(labels[0])
#criando modelo

def model_load(input_shape, output):

    model= Sequential()

    model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=input_shape, use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.01, l2=0.01)))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(kernel_size=(3,3), filters=64, activation='tanh'))

    model.add(MaxPool2D(pool_size=(3,3)))

    

    model.add(Flatten())

    

    model.add(Dense(units=32, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=output, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

    

    return model
#instanciando o modelo

model = model_load(input_shape, output)
model.summary()
#treinando o modelo

model.fit(imagens,labels, batch_size=64, epochs=20, validation_data=(np.array(imagens_test), output_test))
val_loss, val_acc = model.evaluate(np.array(imagens_test), output_test)

print(val_loss, val_acc)