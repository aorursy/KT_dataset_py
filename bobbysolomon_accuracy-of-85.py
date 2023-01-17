from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers

import pandas as p

import numpy as np

from keras.models import Sequential

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator 





datagen = ImageDataGenerator(rescale = 1/255)

train_generator=datagen.flow_from_directory(

directory=r'../input/intel-image-classification/seg_train',

batch_size=64,

class_mode="categorical",

classes=["buildings", "forest", "glacier","mountain","sea","street" ],

target_size=(32,32))





test_generator=datagen.flow_from_directory(

directory=r'../input/intel-image-classification/seg_test',

batch_size=64,

class_mode="categorical",

classes=["buildings", "forest", "glacier","mountain","sea","street" ],

target_size=(32,32))



datagen1=ImageDataGenerator(rescale=1./255.)



test1_generator=datagen1.flow_from_directory(

directory=r'../input/intel-image-classification/seg_pred',

batch_size=64,

class_mode=None,

target_size=(32,32))





model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(32,32,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])                                                          

        

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=test_generator.n//test_generator.batch_size

STEP_SIZE_TEST=test1_generator.n//test1_generator.batch_size

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=test_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=50)



model.evaluate_generator(generator=test_generator,

steps=STEP_SIZE_TEST)





test1_generator.reset()

pred=model.predict_generator(test1_generator,

steps=STEP_SIZE_TEST,

verbose=1)



predicted_class_indices=np.argmax(pred,axis=1)



print(predicted_class_indices)



l=[]



for i in predicted_class_indices:

    if i==0:

        l.append('Buildings')

    elif i==1:

        l.append('forests')

    elif i==2:

        l.append('Glacier')

    elif i==3:

        l.append('Mountains')

    elif i==4:

        l.append("sea")

    else:

        l.append('Streets')

    

print(l)




