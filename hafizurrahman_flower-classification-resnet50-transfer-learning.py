# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import shutil

import matplotlib.pyplot as plt

%matplotlib inline 



from IPython.display import Image, display

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/flowers-recognition/"))

# Any results you write to the current directory are saved as output.
labels = os.listdir("../input/flowers-recognition/flowers/flowers/")
num_classes = len(set(labels))

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# Create model

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



# Do not train first layer (ResNet) as it is already pre-trained

model.layers[0].trainable = False



# Compile model

from tensorflow.python.keras import optimizers



sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
train_folder = '../input/flowers-recognition/flowers/flowers/'



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,

                                    horizontal_flip=True,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    validation_split=0.2)# set validation split



train_generator = data_generator.flow_from_directory(

    train_folder,

    target_size=(image_size, image_size),

    batch_size=24,

    class_mode='categorical',

    subset='training'

    )

validation_generator = data_generator.flow_from_directory(

    train_folder,

    target_size=(image_size, image_size),

    batch_size=24,

    class_mode='categorical',

    subset='validation'

    )
NUM_EPOCHS = 20

EARLY_STOP_PATIENCE = 5
# Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint



cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)

cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5',

                                  monitor = 'val_loss',

                                  save_best_only = True,

                                  mode = 'auto')
import math



fit_history = model.fit_generator(

    train_generator,

    steps_per_epoch=10,

    validation_data=validation_generator,

    validation_steps=10,

    epochs=NUM_EPOCHS,

    callbacks=[cb_checkpointer, cb_early_stopper])

model.load_weights("../working/best.hdf5")
print(fit_history.history.keys())
plt.figure(1, figsize = (15,8)) 

    

plt.subplot(221)  

plt.plot(fit_history.history['acc'])  

plt.plot(fit_history.history['val_acc'])  

plt.title('model accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 

    

plt.subplot(222)  

plt.plot(fit_history.history['loss'])  

plt.plot(fit_history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 



plt.show()