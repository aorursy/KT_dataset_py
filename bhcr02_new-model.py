import numpy as np

import keras

from keras import backend as k

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt

%matplotlib inline

from keras.models import Model

from keras.preprocessing import image

from keras.applications import imagenet_utils

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.preprocessing import image

from keras.models import Model

from keras.models import model_from_json

from keras.layers import Input

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential

#from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

train_path='/kaggle/input/cmaterfi/cmaterdb/train'

valid_path='/kaggle/input/cmaterfi/cmaterdb/valid'

test_path='/kaggle/input/cmaterfi/cmaterdb/test'



def myFunc(image):

    image = np.array(image)

    image= np.expand_dims(image, axis=0)

    image = image.astype('float32')

    image /= 255

    return image



train_datagen = ImageDataGenerator(preprocessing_function=myFunc, validation_split=0.2) # set validation split







train_batches=train_datagen.flow_from_directory(train_path,target_size=(28,28),color_mode='grayscale',batch_size=32,subset='training')

valid_batches= train_datagen.flow_from_directory(train_path,target_size=(28,28),color_mode='grayscale',batch_size=32,subset='validation')

test_batches= ImageDataGenerator(preprocessing_function=myFunc).flow_from_directory(test_path,target_size=(28,28),color_mode='grayscale',batch_size=32,shuffle=False)

#mobile=keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

#mobile.add (Dense(1024,activation='relu'))



#x = mobile.layers[-2].output

#x = Dropout(0.5)(x)



#x= Dense(1024,activation='relu')(x)

#x = Dense(512,activation='relu')(x)

#predictions = Dense(3,activation='softmax')(x)

#model = Model(inputs=mobile.input,outputs=predictions)

#from keras.models import load_model





#train_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,target_size=(224,224),batch_size=32)

#valid_batches= ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path,target_size=(224,224),batch_size=32)

#test_batches= ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path,target_size=(224,224),batch_size=32)

#mobile = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

#mobile = keras.applications.mobilenet.MobileNet(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

#x = mobile.layers[-2].output





#x= Dense(1024,activation='relu')(x)



#x = Dense(512,activation='relu')(x)



model = Sequential()

#, data_format='channels_first'

model.add(Conv2D(8, kernel_size=3, activation='relu',padding='same', input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(8, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(16, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(16, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))

model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(256, kernel_size=3, activation='relu',padding='same'))



model.add(BatchNormalization())

#model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))

model.add(Conv2D(512, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(512, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(1024, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(2048, kernel_size=3, activation='relu'))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dense(2048, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))  

model.add(Dense(800, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))  

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))     

#model.add(Dense(256, activation='relu'))

#model.add(BatchNormalization())

#model.add(Dropout(0.5))     

model.add(Dense(230, activation='softmax'))        

#model.summary()







#from keras.models import load_model

#model = load_model('/kaggle/input/model111/model-050-0.984515-0.969163.h5')

#predictions = Dense(122,activation='softmax')(x)

#model = Model(inputs=mobile.input,outputs=predictions)



#from keras.models import load_model

model = Sequential()

#, data_format='channels_first'

model.add(Conv2D(16, kernel_size=3, activation='relu',padding='same', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())



model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))



model.add(Conv2D(196, kernel_size=3, activation='relu',padding='same'))

model.add(Conv2D(256, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))



model.add(Conv2D(512, kernel_size=3, activation='relu'))

model.add(Conv2D(1024, kernel_size=3, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(2048, kernel_size=3, activation='relu',padding='same'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dense(2048, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))   

model.add(Dense(400, activation='relu'))

model.add(BatchNormalization())

#model.add(Dropout(0.5))

model.add(Dense(231, activation='softmax'))        

model.summary()



model.summary()

checkpoint = keras.callbacks.callbacks.ModelCheckpoint('/kaggle/working/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max') 







model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
from keras.models import load_model

model = load_model('/kaggle/input/model22/model-008-0.981344-0.959634.h5')

model.summary()
his = model.fit_generator(train_batches,steps_per_epoch=1213,epochs =20 ,verbose=1,validation_data = valid_batches,validation_steps=301,callbacks=[checkpoint])





loss, acc = model.evaluate_generator(test_batches, steps=360, verbose=1)

acc
model.save('/kaggle/working/final.h5')