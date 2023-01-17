# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input/asl_alphabet_test/asl_alphabet_test"))

train_dir="../input/asl_alphabet_train/asl_alphabet_train"

test_dir="../input/asl_alphabet_test/"

# Any results you write to the current directory are saved as output.
#print(os.listdir(train_dir+'/A'))

image = cv2.imread(train_dir+'/B/B970.jpg')

plt.imshow(image)

import keras

from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout,SeparableConv2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers

from keras.layers import BatchNormalization

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),input_shape=(128,128,3),activation='relu',strides=1,kernel_initializer='he_normal',padding='same',kernel_regularizer=regularizers.l2(0.01)))

#model.add(Conv2D(32,kernel_size=(3,3),activation='relu',strides=1,kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

#model.add(Dropout(0.25))



model.add(Conv2D(64,kernel_size=(3,3),activation='relu',strides=1,kernel_initializer='he_normal',padding='same',kernel_regularizer=regularizers.l2(0.01)))

#model.add(Conv2D(64,kernel_size=(3,3),activation='relu',strides=1,kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

#model.add(Dropout(0.50))



model.add(Conv2D(128,kernel_size=(3,3),activation='relu',strides=1,kernel_initializer='he_normal',padding='same',kernel_regularizer=regularizers.l2(0.01)))

#model.add(Conv2D(128,kernel_size=(3,3),activation='relu',strides=1,kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Flatten())

#model.add(Dropout(0.5))





model.add(Dense(512,activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))



model.add(Dense(29,activation='softmax',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01)))

model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

train_datagen=ImageDataGenerator( 

                                    validation_split=0.1,rescale=1.0/255

                                )

test_datagen=ImageDataGenerator(rescale=1.0/255

                              )

                                                 



    

train_generator=train_datagen.flow_from_directory(train_dir,

                                               target_size=(128,128),

                                               batch_size=256,

                                               class_mode='categorical',

                                               subset='training',

                                              shuffle=True

                                               )



validation_generator=train_datagen.flow_from_directory(train_dir,

                                               target_size=(128,128),

                                               batch_size=256,

                                               class_mode='categorical',

                                               subset='validation'

                                               )

test_generator=test_datagen.flow_from_directory(test_dir,

                                               target_size=(128,128),

                                               batch_size=20,

                                               class_mode='categorical'

                                                )

    

callbacks=[keras.callbacks.ReduceLROnPlateau(

monitor='val_loss',

factor=0.1,

patience=10)]

          



history=model.fit_generator(train_generator,

                           steps_per_epoch = 305,

                           epochs=10,

                           validation_data=validation_generator,

                            validation_steps=33

                           )
model.save("ANANT.h5")

hist=history.history

print(hist.keys()

     )


val_acc=hist['val_acc']

val_loss=hist['val_loss']

acc=hist['acc']

loss=hist['loss']
import matplotlib.pyplot as plt

epochs=range(1,len(acc)+1)



print(epochs)
plt.plot(epochs,acc,'bo',label='Training_acc')

plt.plot(epochs,val_acc,'b',label='Validation_acc')

plt.title('Training_vs Validation Accuracy')

plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='Training_loss')

plt.plot(epochs,val_loss,'b',label='Validation_loss')

plt.title('Training_vs Validation Loss')

plt.legend()

plt.figure()
from keras.preprocessing.image import load_img,img_to_array

from keras.models import load_model

from keras.applications.vgg16 import preprocess_input

image=load_img("../input/asl_alphabet_test/asl_alphabet_test/C_test.jpg",target_size=(128,128))

image=img_to_array(image)

image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])

image=preprocess_input(image)

print(image.shape)
y_hat=model.predict(image)



labels=['A','B','C','D','E','F','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

i=np.argmax(y_hat)  

print(i)

print("prediction is",labels[i])
print(y_hat)
model.predict_generator(test_generator,steps=25)

                        
from keras.models import load_model
model = load_model('ANANT.H5')
import os

print(os.listdir())