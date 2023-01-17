# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/images/images/train/"))



# Any results you write to the current directory are saved as output.
# displaying some images for different expression

import numpy as np

import seaborn as sns

from keras.preprocessing.image import load_img,img_to_array

import matplotlib.pyplot as plt

import os



# size of the image: 48*48 pixels

pic_size = 48

    

base_path = "../input/images/images/"



plt.figure(0, figsize = (12,20))

cpt =0



for expression in os.listdir(base_path + "train/"):

    for i in range(1,5):

        cpt = cpt + 1

        plt.subplot(7,4,cpt)

        img = load_img(base_path + "train/" + expression + '/' +

                      os.listdir(base_path +"train/" + expression)[i],

        target_size =(pic_size, pic_size))

        plt.imshow(img,cmap = "gray")

plt.tight_layout()

plt.show()

for expression in os.listdir(base_path + "train"):

    print(str(len(os.listdir(base_path+"train/"+expression)))+" "+expression +" " + "images")

    

    
from keras.preprocessing.image import ImageDataGenerator



#  number of training examples utilized in one training

batch_size = 128



training_datagen = ImageDataGenerator()

validation_datagen = ImageDataGenerator()



train_generator = training_datagen.flow_from_directory(base_path + 'train',

                                                       target_size =(pic_size,pic_size),

                                                       color_mode ='grayscale',

                                                       batch_size = batch_size,

                                                       class_mode ='categorical',

                                                       shuffle = True)



validation_generator = validation_datagen.flow_from_directory(base_path + 'validation',

                                                       target_size = (pic_size,pic_size),

                                                       color_mode = 'grayscale',

                                                       batch_size = batch_size,

                                                       class_mode ='categorical',

                                                       shuffle = True)

from keras.layers import Dense, Input, Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D

from keras.models import Model,Sequential

from keras.optimizers import Adam 



# no of possible label values

nb_classes = 7



#initialising the CNN

model = Sequential()



#1-Convolution

model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 2nd Convolution layer

model.add(Conv2D(128,(5,5), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 3rd Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 4th Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#flattening

model.add(Flatten())



# Fully connected layer 1st layer

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



# Fully connected layer 2nd layer

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(nb_classes, activation='softmax'))



opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
%%time



# number of epochs to train the NN

epochs = 50



from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]



history = model.fit_generator(generator=train_generator,

                                steps_per_epoch=train_generator.n//train_generator.batch_size,

                                epochs=epochs,

                                validation_data = validation_generator,

                                validation_steps = validation_generator.n//validation_generator.batch_size,

                                callbacks=callbacks_list

                                )

# serialize model structure to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
# plot the evolution of Loss and Acuracy on the train and validation sets



import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)

plt.suptitle('Optimizer : Adam', fontsize=10)

plt.ylabel('Loss', fontsize=16)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='upper right')



plt.subplot(1, 2, 2)

plt.ylabel('Accuracy', fontsize=16)

plt.plot(history.history['acc'], label='Training Accuracy')

plt.plot(history.history['val_acc'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.show()



