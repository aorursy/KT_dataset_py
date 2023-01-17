import numpy as np

import argparse

import matplotlib.pyplot as plt

import cv2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





warnings.filterwarnings('ignore')





def plot_model_history(model_history):

        """

        Plot Accuracy and Loss curves given the model_history

        """

        fig, axs = plt.subplots(1,2,figsize=(15,5))

        

        axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])

        axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])

        axs[0].set_title('Model Accuracy')

        axs[0].set_ylabel('Accuracy')

        axs[0].set_xlabel('Epoch')

        axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

        axs[0].legend(['train', 'val'], loc='best')

        

        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

        axs[1].set_title('Model Loss')

        axs[1].set_ylabel('Loss')

        axs[1].set_xlabel('Epoch')

        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

        axs[1].legend(['train', 'val'], loc='best')

        fig.savefig('plot.png')

        plt.show()

        return





def emotion():



    train_dir = '../input/fer2013ds/data/train'

    val_dir = '../input/fer2013ds/data/test'



    num_train = 28709

    num_val = 7178

    batch_size = 64

    num_epoch = 70



    train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)



    train_generator = train_datagen.flow_from_directory(

            train_dir,

            target_size=(48,48),

            batch_size=batch_size,

            color_mode="grayscale",

            class_mode='categorical')



    validation_generator = val_datagen.flow_from_directory(

            val_dir,

            target_size=(48,48),

            batch_size=batch_size,

            color_mode="grayscale",

            class_mode='categorical')



    model = Sequential()



    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))



    

    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

    model_info = model.fit_generator(

            train_generator,

            steps_per_epoch=num_train // batch_size,

            epochs=num_epoch,

            validation_data=validation_generator,

            validation_steps=num_val // batch_size)

    plot_model_history(model_info)

    model.save_weights('model.h5')



    return
emotion()