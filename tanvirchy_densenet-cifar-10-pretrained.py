import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
import pandas as pd
from keras.models import Model
from keras import regularizers
import keras
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization
import keras.backend as K
import itertools
%matplotlib inline
from keras.datasets import cifar10
(X_train,y_train),(X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#z-score
mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
X_train = (X_train-mean)/(std+1e-7)
X_test = (X_test-mean)/(std+1e-7)
y_train_one_hot = to_categorical(y_train,10)
y_test_one_hot = to_categorical(y_test,10)
datagen = ImageDataGenerator(rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                           shear_range = 0.2,
                           zoom_range=0.2,
                          horizontal_flip=True)
datagen.fit(X_train)
net = keras.applications.densenet.DenseNet121(include_top=False,input_shape=(32,32,3))
        
net.summary()


x = net.output

x=Flatten()(x)
x=Dense(64, activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.3)(x)
prediction_layer = Dense(10, activation='softmax')(x) 
model = Model(inputs=net.input, outputs=prediction_layer)

model.summary()

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
from keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)

epochs=100
his  = model.fit(X_train,
                y_train_one_hot,
                batch_size=64,
                epochs=100,
                 validation_data=(X_test,y_test),
                callbacks=[es])
score = model.evaluate(X_test,y_test_one_hot)
print('score : ',score[1]*100,'%')
model.save_weights('desnetpreeva82.hdf5')
model.load_weights('desnetcv eva78.hdf5')
img_pred = image.load_img('../input/cifar10/sedan3.jpg', target_size=(32,32,3))
plt.imshow(img_pred)
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

#Get the probabilities
probabilities = model.predict(img_pred)
probabilities
class_name =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

index = np.argsort(probabilities[0,:])
print('Most likely class :', class_name[index[9]] , ', Probability : ', probabilities[0 , index[9]])
print('Most second  likely class :', class_name[index[8]] , ', Probability : ', probabilities[0 , index[8]])
print('Most third likely class :', class_name[index[7]] , ', Probability : ', probabilities[0 , index[7]])
print('Most fourth  likely class :', class_name[index[6]] , ', Probability : ', probabilities[0 , index[6]])
print('Most fifth  likely class :', class_name[index[5]] , ', Probability : ', probabilities[0 , index[5]])