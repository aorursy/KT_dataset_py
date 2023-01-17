# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/Sign-language-digits-dataset"))



# Any results you write to the current directory are saved as output.
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')

Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
import matplotlib.pyplot as plt

img_size = 64





plt.subplot(1,2,1)

plt.imshow(x_l[300].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1,2,2)

plt.imshow(x_l[990].reshape(img_size, img_size))

plt.axis('off')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_l,Y_l,test_size=0.15,random_state=78)

x_train = x_train.reshape(-1,64,64,1) #(64,64,1)

x_test = x_test.reshape(-1,64,64,1)    #(64,64,1)

print("x_train shape : ", x_train.shape)

print("y_train shape : ", y_train.shape)
from sklearn.metrics import confusion_matrix

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam
model = Sequential() #model oluşturmak için kullanılan yapı

#24 adet filter olacak filter'in boyutu (5,5) veri kaybını önlemek için Same aktivasyon relu 

model.add(Conv2D(filters=24, kernel_size= (5,5),padding= 'Same',

                 activation='relu', input_shape = (64,64,1))) 

# (2,2) matrislerdeki en yüksek değeri seç

model.add(MaxPool2D(pool_size=(2,2)))

# nöronların %25 ini kapat rastgelelik artsın diye

model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size= (4,4),padding= 'Same',

                 activation='relu')) 

# (2,2) matrislerdeki en yüksek değeri seç (2,2) lik atlamalar yaparak

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# nöronların %25 ini kapat rastgelelik artsın diye

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(256, activation='relu'))

#model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

#model.add(Dense(128, activation='relu'))

#softmax geliştirilmiş sigmoid fonk sadece binary deil çoklu sınıflandırmada yapabiliyorr

model.add(Dense(10,activation='softmax'))
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=300, validation_data=(x_test,y_test))

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show(1)
import seaborn as sns

# test verisi modele işlenerek çıktılar elde edilir

Y_pred = model.predict(x_test)

# üretilen çıktılar tek boyuta indirgenir 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# olması gereken çıktılar tek boyuta indirgenir

Y_true = np.argmax(y_test,axis = 1) 

# confusion matrix içeriği tanımlanır

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# confusion matrix çizilir

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
