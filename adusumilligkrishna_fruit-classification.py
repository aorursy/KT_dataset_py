import matplotlib.pyplot as plt

import seaborn as sns

import math

import os

import pickle

import numpy as np

from PIL import Image

from keras.preprocessing.image import img_to_array



from keras import applications

from keras import backend as k

from keras import optimizers

from keras.callbacks import ModelCheckpoint,History

from keras.layers import Dense,Input,UpSampling2D,Conv2D

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras_tqdm import TQDMCallback,TQDMNotebookCallback
train_data_dir = '../input/fruits-360_dataset/fruits-360/Training/'

validation_data_dir = '../input/fruits-360_dataset/fruits-360/Test/'

img_width, img_height = 100, 100

batch_size = 64

nb_epoch = 20

nb_channels= 3
img = plt.imread('../input/fruits-360_dataset/fruits-360/Training/Banana/50_100.jpg')

plt.imshow(img)
encoding_dim = 256

def flattened_generator(generator):

    for batch in generator:

        yield (batch.reshape(-1,img_width*img_height*nb_channels), batch.reshape(-1,img_width*img_height*nb_channels))
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode=None, shuffle=True)



validation_generator = test_datagen.flow_from_directory(

        validation_data_dir,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode=None, shuffle=True)
def AE_FC():



    # this is our input layer

    input_img = Input(shape=(img_height*img_width*nb_channels,))

    

    # this is the bottleneck vector

    encoded = Dense(encoding_dim, activation='relu')(input_img)

    

    # this is the decoded layer, with the same shape as the input

    decoded = Dense(img_height*img_width*nb_channels, activation='sigmoid')(encoded)

    

    return Model(input_img, decoded)
autoencoder = AE_FC()

autoencoder.summary()

checkpoint_cnn = ModelCheckpoint(filepath = "model_weights_ae_cnn.h5", save_best_only=True,monitor="val_loss", mode="min" )

history_cnn = History()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

checkpoint = ModelCheckpoint(filepath = "model_weights_ae_fc.h5", save_best_only=True,monitor="val_loss", mode="min" )

history = History()
autoencoder.fit_generator(

    flattened_generator(train_generator),

    samples_per_epoch=math.floor(41322  / batch_size),

    nb_epoch=nb_epoch,

    validation_data=flattened_generator(validation_generator),

    nb_val_samples=math.floor(13877  / batch_size),

    verbose=0,

    callbacks=[history, checkpoint, TQDMNotebookCallback(leave_inner=True, leave_outer=True)])
x_test = validation_generator.next()
decoded_imgs = autoencoder.predict(x_test.reshape(64,30000))

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i+1)

    plt.imshow(x_test[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + n+1)

    plt.imshow(decoded_imgs[i].reshape(100,100,3))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
plt.figure(figsize=(15,10))

plt.plot(history.epoch,history.history["val_loss"])

plt.plot(history.epoch,history.history["loss"])

plt.title("Validation loss and loss per epoch",fontsize=18)

plt.xlabel("epoch",fontsize=18)

plt.ylabel("loss",fontsize=18)

plt.legend(['Validation Loss','Training Loss'],fontsize=14)

plt.show()
with open("history_fc.pickle","wb") as file:

    pickle.dump(history.history, file)