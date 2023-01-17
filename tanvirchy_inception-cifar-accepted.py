import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD
from keras.metrics import categorical_crossentropy
import pandas as pd
from keras.models import Model
from keras import regularizers
import keras,pickle
import os
from keras.layers import Input
import tensorflow as tf
from keras.models import Sequential,load_model
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Activation,Flatten,MaxPool2D,Conv2D,Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import itertools
%matplotlib inline
from keras.datasets import cifar10
(X_train,y_train),(X_test, y_test) = cifar10.load_data()

y_train_one_hot = to_categorical(y_train,10)
y_test_one_hot = to_categorical(y_test,10)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)

it_train = datagen.flow(X_train, y_train_one_hot, batch_size=64)
model = load_model('../input/incepfiles/inceptionc.h5')
model.summary()
model.load_weights('../input/incepfiles/inceptionc.hdf5')
f=open('../input/incepfiles/inceptionchisto.pckl','rb')
history  =  pickle.load(f)
f.close()
#model is unchanged
input_img = Input(shape = (32,32,3))
incp1 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
incp1 = Conv2D(64,(3,3), padding='same',activation='relu')(incp1)

incp2 = Conv2D(64,(1,1), padding='same', activation='relu')(input_img)
incp2 = Conv2D(64,(5,5),padding='same', activation='relu')(incp2)

incp3 = MaxPool2D((3,3),  padding='same',strides=(1,1))(input_img)
incp3 = Conv2D(64,(1,1),padding='same', activation='relu')(incp3)

output = keras.layers.concatenate([incp1, incp2, incp3], axis=3)

output = Flatten()(output)
output = Dense(512, activation='relu')(output)
output = BatchNormalization()(output)
output = Dense(256, activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(256, activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.3)(output)
output = Dense(128, activation='relu')(output)
output = Dense(128, activation='relu')(output)
output = Dropout(0.3)(output)
output = Dense(128, activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.4)(output)
output = Dense(64, activation='relu')(output)
output = Dense(10, activation='softmax')(output)
model = Model(inputs=input_img, outputs=output)
model.summary()
model.compile(Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
x=model.evaluate(X_test,y_test_one_hot)
x
plt.plot(history['loss'])
plt.plot(history['val_loss'])

plt.title('Loss')
plt.legend(['train', 'validation'])
plt.show()
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])

plt.title('Accuracy')
plt.legend(['train', 'validation'])
plt.show()
bst_val_score = max(history['val_accuracy'])
bst_val_score
img_pred = image.load_img('../input/cifar10/horse4.jpg', target_size=(32,32,3))
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


