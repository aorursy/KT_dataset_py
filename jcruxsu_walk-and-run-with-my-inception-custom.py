import os,shutil

import glob

import numpy as np

import cv2

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.models import *

from keras.layers import *

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.preprocessing.image import *

from keras.callbacks import EarlyStopping

from keras import regularizers,optimizers

from keras.callbacks import LearningRateScheduler

from keras import *

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

#walk is 1, run is 0

original_dataset_dir = "../input"



train_dir = os.path.join(original_dataset_dir,'walk_or_run_train/train')

#'../input/walk_or_run_train/train'

test_dir= os.path.join(original_dataset_dir,'walk_or_run_test/test')

#'../input/walk_or_run_test/test'

def lr_schedule(epoch):

    lrate = 0.0005

    if epoch < 2:

        lrate = 0.003

    if epoch > 5:

        lrate = 0.0001

    return lrate
es = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
train_datagen = ImageDataGenerator(rescale=1/255,rotation_range=30,width_shift_range=0.1,\

                             height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,\

                                 horizontal_flip=True,vertical_flip=False)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=16,

    class_mode='categorical',

    seed=2019,

    color_mode='rgb'

)

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(224, 224),

    batch_size=16,

    class_mode='categorical',color_mode='rgb')
weight_decay=1e-4

img_input = Input(shape=(224,224,3))

conv_base =Conv2D(32,(2,2),kernel_regularizer=regularizers.l2(weight_decay),activation='elu',padding='same')(img_input)

conv_base =Conv2D(16,(3,3),kernel_regularizer=regularizers.l2(weight_decay),activation='elu',padding='same')(conv_base)

conv_base =Conv2D(16,(4,4),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_base)

conv_base = BatchNormalization()(conv_base)





conv_layer_1 =Conv2D(16,(2,2),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_base)

conv_layer_1 = ZeroPadding2D((1,1))(conv_layer_1)

conv_layer_1 =Conv2D(32,(2,2),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_layer_1)

conv_layer_1 = BatchNormalization()(conv_layer_1)



conv_layer_2 =Conv2D(32,(3,3),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_base)

conv_layer_2 = ZeroPadding2D((2,2))(conv_layer_2)

conv_layer_2 =Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_layer_2)

conv_layer_2 = BatchNormalization()(conv_layer_2)







conv_layer_3 =Conv2D(8,(1,1),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_base)

conv_layer_3 =Conv2D(16,(1,1),kernel_regularizer=regularizers.l2(weight_decay),activation='elu')(conv_layer_3)

conv_layer_3 = BatchNormalization()(conv_layer_3)



conv_final = concatenate([conv_layer_1,conv_layer_2,conv_layer_3])



gap = GlobalAveragePooling2D()(conv_final)

Den = Dense(16,activation='relu')(gap)

Den = BatchNormalization()(Den)

Den = Dense(2,activation='sigmoid')(Den)



model = Model(img_input,Den)
model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=0.0005, decay=0.00005), metrics=['accuracy'])

model.summary()
preds = []

ensemble_call_back =[LearningRateScheduler(lr_schedule),es]

temp_model = model

model.fit_generator(train_generator,steps_per_epoch=300,epochs=10,verbose=1,validation_data=test_generator,validation_steps=100)

preds.append( temp_model.predict_generator(test_generator,16))
#acc = history.history['acc']

#val_acc = history.history['val_acc']

#loss= history.history['loss']

#val_loss = history.history['val_loss']

res =[]



#epochs = range(1,len(acc)+1)
#plt.plot(epochs,acc,'bo',label='Train_accuracy')

#plt.plot(epochs,val_acc,'b',label='validation_accuracy')

#plt.title('Training and validation accuracy')

#plt.legend()

#plt.figure()



#plt.plot(epochs,loss,'bo',label='Train_loss')

#plt.plot(epochs,val_loss,'b',label='validation_loss')

#plt.title('Training and validation loss')

#plt.legend()





#plt.show()







#test = model.evaluate_generator(test_generator)

#print('validation acc-',test[1]*100,'%')