import os, cv2, random

import keras

from keras.models import Sequential

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers



import numpy as np

import PIL

import scipy

import pandas as pd

import h5py



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns



from tqdm import tqdm

%matplotlib inline 



seed = 7

np.random.seed(seed)
TRAIN_DIR = '../input/train'

TEST_DIR = '../input/test'



CHANNELS = 3



IMG_SIZE = 50

#LR = 0.001





dog_images = [i for i in os.listdir(TRAIN_DIR) if i.split('.')[-3]=='dog']

cat_images = [i for i in os.listdir(TRAIN_DIR) if i.split('.')[-3]=='cat']

print(dog_images[:10])

train_input = dog_images[:1000] + cat_images[:1000]

test_images = dog_images[-500:] + cat_images[-500:]



#test_images = [i for i in os.listdir(TEST_DIR)[:1000]]

print(test_images[-10:])



    
#Feeding only 1000 Dogs and 1000 Cats images into model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def label_images(img):

    word_label = img.split('.')[-3]

    if word_label == 'cat': return [0, 1]

    elif word_label == 'dog': return [1, 0]

    

def create_train_data():

    training_data = []

    for img in tqdm(train_input):

    #for img in tqdm(os.listdir(TRAIN_DIR)):

        label = label_images(img)

        path = os.path.join(TRAIN_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE ))

        training_data.append([np.array(img), np.array(label)])

    np.save('training_data.npy', training_data)

        

    return training_data



       #img = img.reshape(img.shape[0], 3, 50, 50).astype('float32')

             

def processing_test_data():

    test_data = []

    for img in tqdm(test_images):

        label = label_images(img)

        path = os.path.join(TRAIN_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        test_data.append([np.array(img), np.array(label)])

        

        

    np.save('testing_data.npy', test_data)

    return test_data



train_data = create_train_data()

test_data = processing_test_data()
train = train_data

test = test_data

#print(test_data[:5])

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

Y = np.array([i[1] for i in train])

#print(Y.shape)

print("X_Shape", X.shape)

print("Y_shape", Y.shape)

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

Y_test = np.array([i[1] for i in test])

print("X_test_shape", X_test.shape)

print("Y_test_shape", Y_test.shape)



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



basefilter = 32

lr = 0.0001



model = Sequential()

model.add(Conv2D(basefilter, (3,3), padding='same',

                 input_shape=(X.shape[1:])))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(basefilter, (3,3), padding='same',

                 input_shape=(X.shape[1:])))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(basefilter*2, (3,3), padding='same',

                 input_shape=(X.shape[1:])))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(basefilter*2, (3,3), padding='same',

                 input_shape=(X.shape[1:])))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))





model.add(Conv2D(basefilter*4, (3,3), padding='same',

                 input_shape=(X.shape[1:])))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(basefilter*4, (3,3), padding='same',

                 input_shape=(X.shape[1:])))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))



model.add(Flatten())



model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation('sigmoid'))



model.summary()

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=40, 

                             zoom_range=0.2,

                             shear_range=0.2,

                             horizontal_flip=True,)



datagen.fit(X)

for X_batch, y_batch in datagen.flow(X, Y):

    X_batch = X_batch[:5]

    

    for img in range(0,5):

        plt.figure(figsize=(5,5))

        plt.imshow(X_batch[img])

        plt.show()

    break

    

#Model Training

batch_size = 100

epochs = 25



rms_opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



model.compile(loss='binary_crossentropy',

              optimizer=rms_opt,

              metrics=['accuracy'])

    

model.fit_generator(datagen.flow(X, Y, 

                                 batch_size=batch_size),

                                 steps_per_epoch=X.shape[0] // batch_size, 

                                 epochs = 2*epochs,

                                 verbose=1, validation_data=(X_test, Y_test))

model.save_weights('dog_cats_img_augument_50.h5')





scores = model.evaluate(X_test, Y_test, verbose=0)

print("Accuracy_50: %.2f%%" % (scores[1]*100))



rms_opt = keras.optimizers.rmsprop(lr=0.0003, decay=1e-6)



model.compile(loss='binary_crossentropy',

              optimizer=rms_opt,

              metrics=['accuracy'])

    

model.fit_generator(datagen.flow(X, Y, 

                                 batch_size=batch_size),

                                 steps_per_epoch=X.shape[0] // batch_size, 

                                 epochs = 3*epochs,

                                 verbose=1, validation_data=(X_test, Y_test))

model.save_weights('dog_cats_img_augument_75.h5')





scores = model.evaluate(X_test, Y_test, verbose=0)

print("Accuracy_75: %.2f%%" % (scores[1]*100))



rms_opt = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)



model.compile(loss='binary_crossentropy',

              optimizer=rms_opt,

              metrics=['accuracy'])

    

model.fit_generator(datagen.flow(X, Y, 

                                 batch_size=batch_size),

                                 steps_per_epoch=X.shape[0] // batch_size, 

                                 epochs = 5*epochs,

                                 verbose=1, validation_data=(X_test, Y_test))

model.save_weights('dog_cats_img_augument_125.h5')





scores = model.evaluate(X_test, Y_test, verbose=0)

print("Accuracy_125: %.2f%%" % (scores[1]*100))