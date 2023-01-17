## The idea of this kernel is to prepare a CNN to convert images to vec, so you can use it as feature in any ML model.

## To apply this to your model you have to copy the model and load the weights 'autoencoder.h5'
import os, sys

import numpy as np

import pandas as pd



import keras

from keras.preprocessing.image import load_img, img_to_array

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from keras.models import Model

from keras_preprocessing.image import ImageDataGenerator

from keras.optimizers import Adadelta

from keras.preprocessing import image

import matplotlib.pyplot as plt

from copy import deepcopy
path_train = '../input/train_images/'

path_test = '../input/test_images/'



img_paths = [x for x in os.listdir(path_train)]# + [x for x in os.listdir(path_test)]

print(len(img_paths))

img_paths = pd.DataFrame(img_paths, columns=['paths'])

img_paths.head()
img_paths[img_paths['paths'] == 'train_images/9ede28a9d-1.jpg']
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.1)



img_width = 64

img_height = 64



train_generator=datagen.flow_from_dataframe(

                            dataframe=img_paths,

                            directory=path_train,

                            x_col="paths",

                            y_col="paths",

                            subset="training",

                            batch_size=32,

                            seed=42,

                            shuffle=True,

                            class_mode="input",

                            target_size=(img_width, img_height))



valid_generator=datagen.flow_from_dataframe(

                            dataframe=img_paths,

                            directory=path_train,

                            x_col="paths",

                            y_col="paths",

                            subset="validation",

                            batch_size=32,

                            seed=42,

                            shuffle=True,

                            class_mode="input",

                            target_size=(img_width, img_height))
input_img = Input(shape=(img_width, img_height, 3))



x = Convolution2D(8, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Convolution2D(32, (2, 2), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Convolution2D(64, (2, 2), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)



x = Flatten()(x)

x = Dense(300, activation='relu')(x)

x = Dense(64 * 4 * 4, activation='relu')(x)

x = Reshape((4,4,64))(x)



x = Convolution2D(64, (2, 2), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Convolution2D(32, (2, 2), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)



decoded = Convolution2D(3, (3, 3), activation='sigmoid', padding='same')(x)



autoencoder = Model(input_img, decoded)

# autoencoder.compile(optimizer=Adadelta(lr=0.01), loss='binary_crossentropy')

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder.fit_generator(generator=train_generator,

                    steps_per_epoch=32,

                    #validation_data=valid_generator,

                    #validation_steps=32,

                    epochs=3000)
autoencoder.save_weights('autoencoder.h5')
def load_image(path):

    img = load_img(path, target_size=(img_width, img_height)) 

    img = np.asarray(img).reshape((1, img_width, img_height, 3)) / 255

    return img



img_test = [path_test + x for x in os.listdir(path_test)]



img_h = np.zeros((1,img_height*5*2+1,3))

for i in range(50):

    img_v = np.zeros((img_height,1,3))

    for j in range(5):

        ix = j+i*5

        img = load_image(img_test[ix])

        dec = autoencoder.predict(img) # Decoded image

        img = img[0]

        dec = dec[0]

        img_v = np.hstack((img_v, img, dec))

    img_h = np.vstack((img_h, img_v))

plt.figure(figsize=(16,100))    

plt.imshow(img_h)

plt.axis('off')

plt.show()
## Here is the transformation of a image in a 20 dimensions vector feature :)

feat_extractor = Model(autoencoder.inputs, autoencoder.layers[-12].output)

#feat_extractor.summary()

feat_extractor.predict(load_image(img_test[0]))