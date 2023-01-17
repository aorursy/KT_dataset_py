import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.metrics import categorical_crossentropy
import pandas as pd
from keras.models import Model
from keras import regularizers
import keras
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


it_train = datagen.flow(X_train, y_train_one_hot, batch_size=128)
model = load_model('../input/savedfiles/inception.h5')
model.summary()
model.load_weights('../input/savedfiles/inception.hdf5')

#unchanged
input_img = Input(shape = (32,32,3))

incp1 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
incp1 = Conv2D(64,(3,3), padding='same',activation='relu')(incp1)

incp2 = Conv2D(64,(1,1), padding='same', activation='relu')(input_img)
incp2 = Conv2D(64,(5,5),padding='same', activation='relu')(incp2)

incp3 = MaxPool2D((3,3),  padding='same',strides=(1,1))(input_img)
incp3 = Conv2D(64,(1,1),padding='same', activation='relu')(incp3)

output = keras.layers.concatenate([incp1, incp2, incp3], axis=3)

output = Flatten()(output)
output = Dense(128, activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(128, activation='relu')(output)
output = Dropout(0.3)(output)
output = Dense(128, activation='relu')(output)
output = Dropout(0.4)(output)

output = Dense(10, activation='sigmoid')(output)

model = Model(inputs=input_img, outputs=output)  
model.summary()
for layer in model.layers[:-8]:
    layer.trainable = False

model.summary()
       

opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


es= EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=8)
filepath = "inception_updated.h5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
steps = int(X_train.shape[0] / 128)

epochs = 100

hist  = model.fit_generator(it_train,
                 steps_per_epoch=steps,
                 epochs=epochs,
                validation_data=(X_test,y_test_one_hot),
                 callbacks=[es,checkpoint])
x=model.evaluate(X_test,y_test_one_hot)
x
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
print('Most fourth  likely class :', class_name[index[6]] , ', Probability : ', probabilities[0 , index[6]])
print('Most fifth  likely class :', class_name[index[5]] , ', Probability : ', probabilities[0 , index[5]])

