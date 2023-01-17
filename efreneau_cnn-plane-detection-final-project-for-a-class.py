import os

import math

import sklearn

import collections

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Activation, Dense, Flatten, Dropout

from keras.layers.normalization import BatchNormalization

from keras.callbacks import LearningRateScheduler



#Swish

from keras import backend as K

from keras.utils.generic_utils import get_custom_objects
def LoadDir(dirname):

    imgs = [np.zeros([20,20,3])]

    labels = [0]

    for imgname in os.listdir(dirname):

        

        if(imgname[0]=='1'):

            labels.append(1)

        else:

            labels.append(0)

        

        img = Image.open(os.path.join(dirname, imgname))

        img = np.array(img)

        imgs.append(img)

        

    return np.array(imgs),np.array(labels)

imgs, labels = LoadDir(r'../input/planesnet/planesnet')
x_train, x_temp, y_train, y_temp = train_test_split(imgs,labels,test_size=0.20, random_state=42,shuffle=True)

x_valid, x_test, y_valid, y_test = train_test_split(x_temp,y_temp,test_size=0.50, random_state=42,shuffle=True)
x_train1 = np.rot90(x_train,k=1,axes=(1,2))#Generate new samples

x_train2 = np.rot90(x_train,k=2,axes=(1,2))

x_train3 = np.rot90(x_train,k=3,axes=(1,2))



x_train = np.append(x_train,x_train1,axis=0)

x_train = np.append(x_train,x_train2,axis=0)

x_train = np.append(x_train,x_train3,axis=0)



y_train = np.append(y_train,y_train)

y_train = np.append(y_train,y_train)



print(x_train.shape)
#https://arxiv.org/pdf/1710.05941.pdf

model = Sequential()



def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Activation(swish)})



model.add(Conv2D(16, 2, padding='valid',input_shape=(20,20,3)))

model.add(BatchNormalization())

model.add(Activation('tanh'))

#model.add(MaxPooling2D((2,2)))



model.add(Conv2D(32, 3, padding='valid'))

model.add(BatchNormalization())

model.add(Activation('swish'))

#model.add(Dropout(0.2))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(64, 2, kernel_initializer='random_uniform', padding='valid'))#64 2

model.add(BatchNormalization())

model.add(Activation('swish'))

#model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))





opt = keras.optimizers.SGD(lr=0.01, momentum=0.1, decay=0.00005, nesterov=True)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])

print(model.summary())
log = model.fit(x_train, y_train, shuffle=True, validation_data=(x_valid, y_valid), epochs=50)
y_pred = model.predict(x_test)

y_pred[y_pred>0.5] = 1

y_pred[y_pred<0.5] = 0

y_pred=np.asarray(y_pred, dtype=bool)

target_names = ['No Plane', 'Plane']

print(classification_report(y_test, y_pred, target_names=target_names))

print('Accuracy:',accuracy_score(y_test, y_pred))
#Training loss vs validation loss

tloss = log.history['loss']

vloss = log.history['val_loss']

plt.plot(np.linspace(0,len(tloss)-1,len(tloss)),tloss, label='Training Loss')

plt.plot(np.linspace(0,len(vloss)-1,len(vloss)),vloss, label='Validation Loss')

plt.title('Binary Crossentropy (loss)')

plt.legend()
#Training accuracy vs validation accuracy

tac = log.history['binary_accuracy']

vac = log.history['val_binary_accuracy']

plt.plot(np.linspace(0,len(tac)-1,len(tac)),tac, label='Training Accuracy')

plt.plot(np.linspace(0,len(vac)-1,len(vac)),vac, label='Validation Accuracy')

plt.title('Relative Accuracy')

plt.legend()