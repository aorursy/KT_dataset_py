import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import pandas as pd
from keras.models import Model
from keras import regularizers
import keras
import os
from keras.layers import Input
import tensorflow as tf
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Activation,Flatten,MaxPool2D,Conv2D,Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import itertools
%matplotlib inline
from keras.datasets import cifar10
(X_train,y_train),(X_test, y_test) = cifar10.load_data()


"""datagen = ImageDataGenerator(rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                           shear_range = 0.2,
                           zoom_range=0.2,
                          horizontal_flip=True)
datagen.fit(X_train)"""
y_train_one_hot = to_categorical(y_train,10)
y_test_one_hot = to_categorical(y_test,10)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
plt.imshow(X_train[5])
y_train[5]
input_img = Input(shape = (32,32,3))
incp1 = Conv2D(64,(1,1), padding='same', activation='relu')(input_img)
incp1 = Conv2D(64,(3,3), padding='same', activation='relu')(incp1)

incp2 = Conv2D(64,(1,1), padding='same', activation='relu')(input_img)
incp2 = Conv2D(64,(5,5), padding='same', activation='relu')(incp2)

incp3 = MaxPool2D((3,3), padding='same', strides=(1,1))(input_img)
incp3 = Conv2D(64,(1,1), padding='same', activation='relu')(incp3)

output = keras.layers.concatenate([incp1, incp2, incp3], axis=3)

output = Flatten()(output)
"""output = Dense(128, activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(64, activation='relu')(output)"""

output = Dropout(0.5)(output)
output = Dropout(0.5)(output)
"""output = Dropout(0.5)(output)
output = Dropout(0.3)(output)
output = Dropout(0.3)(output)"""

output = Dropout(0.5)(output)


output = Dense(10, activation='softmax')(output)


model = Model(inputs=input_img, outputs=output)  
model.summary()
model.load_weights('inception88.hdf5')
from keras.optimizers import SGD

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['accuracy'])
from keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50)

epochs=20
his  = model.fit(X_train,
                y_train_one_hot,
                 validation_split=0.3,
                batch_size=128,
                epochs=epochs
                )
score = model.evaluate(X_test,y_test_one_hot)
print('score : ',score[1]*100,'%')
#Accuracy

epoch_nums = range(1, epochs+1)
training_acc = his.history["accuracy"]
validation_acc = his.history["val_accuracy"]
plt.plot(epoch_nums , training_acc)
plt.plot(epoch_nums , validation_acc)
plt.xlabel('epoch')
plt.ylabel('acc ')
plt.legend(['training','validation'], loc='upper right')
plt.show()

epoch_nums = range(1, epochs+1)
training_loss = his.history["loss"]
validation_loss = his.history["val_loss"]
plt.plot(epoch_nums , training_loss)
plt.plot(epoch_nums , validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training','validation'], loc='upper right')
plt.show()
model.save_weights('inception88.hdf5')
img_pred = image.load_img('../input/cifar10/frog4.jpg', target_size=(32,32,3))
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
print('Most third  likely class :', class_name[index[7]] , ', Probability : ', probabilities[0 , index[7]])
