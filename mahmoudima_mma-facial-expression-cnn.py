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
import matplotlib as plt

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D,MaxPooling2D,BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.metrics import confusion_matrix
BATCH_SIZE = 128



def create_model(x=None):

    # we initialize the model

    model = Sequential()



    # Conv Block 1

    model.add(Conv2D(64, (3, 3), input_shape=(48,48,3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3),   padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



    # Conv Block 2

    model.add(Conv2D(128, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



    # Conv Block 3

    model.add(Conv2D(256, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

   

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



    # Conv Block 4

    model.add(Conv2D(512, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

   

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



    # Conv Block 5

    model.add(Conv2D(512, (3, 3),  padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

   

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))





    # FC layers

    model.add(Flatten())

    model.add(Dropout(0.5))

    

    model.add(Dense(7, activation='softmax'))



    return model



model = create_model()



model.summary()
def image_data_generator(data_dir,

                         data_augment=False,

                         batch_size=BATCH_SIZE,

                         target_size=(48, 48),

                         color_mode='rgb',

                         class_mode='categorical',

                         shuffle=True):

    if data_augment:

        datagen = ImageDataGenerator(rescale=1./255,

                                     rotation_range=20,

                                     width_shift_range=0.2,

                                     height_shift_range=0.2,

                                     shear_range=0.2,

                                     zoom_range=0.2,

                                     horizontal_flip=True)

    else:

        datagen = ImageDataGenerator(rescale=1./255)



    generator = datagen.flow_from_directory(data_dir,

                                            target_size=target_size,

                                            color_mode=color_mode,

                                            batch_size=batch_size,

                                            shuffle=shuffle,

                                            class_mode=class_mode)

    return generator
train_generator = image_data_generator('/kaggle/input/mma-facial-expression/MMAFEDB/train',data_augment=True)

validation_generator = image_data_generator('/kaggle/input/mma-facial-expression/MMAFEDB/valid')
model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
import tensorflow.keras

callbacks_list = [

tensorflow.keras.callbacks.EarlyStopping(

monitor='val_accuracy', min_delta=0.0001, 

patience=10, verbose=1, mode='auto',

baseline=None, restore_best_weights=True),

tensorflow.keras.callbacks.ReduceLROnPlateau(

monitor='val_accuracy',

factor=0.5,

patience=5,

verbose=1,

mode='auto')

]
history = model.fit_generator(train_generator,steps_per_epoch=92968//BATCH_SIZE,epochs=100,

                              validation_data=validation_generator,

                              validation_steps=17356//BATCH_SIZE,callbacks=callbacks_list )
def evaluate_model(model=None, filepath=None):

    """return the evaluate """

    if not model:

        assert(filepath)

        model = models.load_model(filepath)

    test_generator = image_data_generator('/kaggle/input/mma-facial-expression/MMAFEDB/test', batch_size=1, shuffle=False)



    nb_samples = len(test_generator)

    score = model.evaluate_generator(test_generator, steps=nb_samples)

    print(score)

    return score
score = evaluate_model(model)
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.xlabel('accuracy')

plt.ylabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.xlabel('loss')

plt.ylabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()