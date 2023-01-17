from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3, activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())                                                                                                                 
classifier.add(Dense(output_dim = 128, activation='relu'))

classifier.add(Dense(output_dim = 1, activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_set = train_datagen.flow_from_directory(

        '../input/dataset/training_set/',

        target_size=(64, 64),

        batch_size=80,

        class_mode='binary')



test_set = test_datagen.flow_from_directory(

        '../input/dataset/test_set/',

        target_size=(64, 64),

        batch_size=20,

        class_mode='binary')

classifier.fit_generator(

        train_set,

        steps_per_epoch=100,

        epochs=25,

        validation_data=test_set,

        validation_steps=100)
classifier.save('model.v1')
from keras.models import load_model

import cv2

import numpy as np



model = load_model('model.v1')



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



img = cv2.imread('../input/dataset/single_prediction/cat_or_dog_2.jpg')

img = cv2.resize(img,(64,64))

img = np.reshape(img,[1,64,64,3])



classes = model.predict_classes(img)

if classes == 0:

    print ('cat') 

if classes == 1:

    print('dog')