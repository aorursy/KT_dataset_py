import numpy as np

import pandas as pd

import seaborn as sns

import os

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.image import load_img,img_to_array
pic_size = 48

b_path = "../input/face-expression-recognition-dataset/images/"
plt.figure(0,figsize=(20,20))

cpt=0

for expression in os.listdir(b_path + "train/"):

    for i in range(1,8):

        cpt += 1

        plt.subplot(7,8,cpt)

        img=load_img(b_path+"train/"+expression+"/"+os.listdir(b_path+"train/"+expression)[i],target_size=(pic_size,pic_size))

        plt.imshow(img,cmap='gray')

        plt.xlabel(os.listdir(b_path+"train/"+expression)[i])

plt.tight_layout()

plt.show()
for expression in os.listdir(b_path + "train"):

    print(str(len(os.listdir(b_path + "train/" + expression)))+" "+expression+" images")
from keras.preprocessing.image import ImageDataGenerator as IDG
train_dat = IDG()

val_dat = IDG()

batch_size = 128

train_gen = train_dat.flow_from_directory(b_path+"train",target_size=(pic_size,pic_size)

                                          ,color_mode="grayscale",batch_size=batch_size,

                                          class_mode="categorical",shuffle=True)

val_gen = val_dat.flow_from_directory(b_path+"validation",target_size=(pic_size,pic_size),

                                       color_mode="grayscale",batch_size=batch_size,

                                       class_mode="categorical",shuffle=False)
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D

from keras.models import Model,Sequential

from keras.optimizers import Adam,SGD,RMSprop
n_classes = 7

#layer 1

model = Sequential()

model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#layer 2

model.add(Conv2D(128,(5,5),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#layer 3

model.add(Conv2D(512,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#layer 4

model.add(Conv2D(512,(3,3),padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

#FC Layer 1

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

#FC Layer 2

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(n_classes,activation='softmax'))

opt = Adam(lr = 0.0001)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
epochs = 48

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5",monitor = "val_acc",verbose=1,

                             save_best_only=True,mode = "max")

callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_gen,steps_per_epoch=train_gen.n//train_gen.batch_size

                              ,epochs=epochs,validation_data = val_gen,validation_steps=val_gen.n//val_gen.batch_size,

                             callbacks = callbacks_list)
model_json = model.to_json()

with open("model.json","w") as json_file:

    json_file.write(model_json)
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

plt.suptitle('Adam Optimizer',fontsize=20)

plt.plot(history.history['loss'],label='Training Loss')

plt.plot(history.history['val_loss'],label='Validation Loss')

plt.legend(loc='upper right')

plt.subplot(1,2,2)

plt.ylabel("Accuracy",fontsize=16)

plt.plot(history.history['accuracy'],label='Training Accuracy')

plt.plot(history.history['val_accuracy'],label = 'Validation Accuracy')

plt.legend()

plt.show()