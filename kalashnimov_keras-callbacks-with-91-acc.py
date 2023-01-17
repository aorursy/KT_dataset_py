# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, BatchNormalization

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers, optimizers

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
#cifar10 = tf.keras.datasets.cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# number of classes

K = len(set(y_train.flatten()))

print("number of classes:", K)
X_train = X_train.astype('float32')/255

X_test = X_test.astype('float32')/255

y_train_cat = to_categorical(y_train, 10)

y_test_cat = to_categorical(y_test, 10)

print("x_train.shape:", X_train.shape)

print("y_train_cat.shape", y_train.shape)
rlr = ReduceLROnPlateau(monitor='val_accuracy', mode ='max', factor=0.5, min_lr=1e-7, verbose = 1, patience=10)

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose = 1, patience=50, restore_best_weights=True)

mc = ModelCheckpoint('cnn_best_model.h5', monitor='val_accuracy', mode='max', verbose = 1, save_best_only=True)

callback_list = [rlr, es, mc]
# build model

def build_model(lr = 0, dc = 0, dr = 0):

    model = Sequential(name = 'CNN_cifar10')

    model.add(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same', input_shape=(32, 32, 3)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.3))



    model.add(Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.4))



    model.add(Flatten())

    model.add(Dense(256, activation='elu', kernel_initializer='he_uniform'))

    model.add(BatchNormalization())

    model.add(Dropout(dr))

    model.add(Dense(10, activation='softmax'))

    # compile model

    opt = optimizers.Adam(lr = lr, decay = dc)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
model = build_model(lr = 0.001, dc = 1e-5, dr = 0.5)
model.summary()
#data augmentation

datagen = ImageDataGenerator(

                            rotation_range=15,

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            horizontal_flip=True,

                            vertical_flip=False

                            )

datagen.fit(X_train)
# run model

model.fit_generator(datagen.flow(X_train, y_train_cat, batch_size = 64),

                                 validation_data = (X_test, y_test_cat),

                                 steps_per_epoch = X_train.shape[0] // 64, 

                                 epochs = 40, verbose = 1,

                                 callbacks = callback_list)
def plot_model(history): 

    fig, axs = plt.subplots(1,2,figsize=(16,5)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy'], 'c') 

    axs[0].plot(history.history['val_accuracy'],'m') 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss'], 'c') 

    axs[1].plot(history.history['val_loss'], 'm') 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper right')

    plt.show()
plot_model(model.history)
saved_model = load_model('cnn_best_model.h5')

test_loss, test_acc = saved_model.evaluate(X_test, y_test_cat, verbose=0)

print('Test Accuracy:', round(test_acc, 2))
y_pred = saved_model.predict_classes(X_test, verbose=0)



class_names = ['airplane',

                'automobile',

                'bird',

                'cat',

                'deer',

                'dog',

                'frog',

                'horse',

                'ship',

                'truck']

y_pred = [class_names[i] for i in y_pred]



submissions=pd.DataFrame({"id": list(range(1, len(y_pred)+1)),

                          "label": y_pred})



submissions.to_csv("submission.csv", index=False)