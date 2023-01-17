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
import os

import warnings               

warnings.filterwarnings('ignore')



import tensorflow as tf

import shutil

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import call


import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam
PATH="/kaggle/input/brain-tumor-classification-mri/"
DATA_PATH = os.path.join(PATH, 'Training')

data_dir_list =np.sort(os.listdir(DATA_PATH))

data_dir_list
DATA_PATH_Test = os.path.join(PATH, 'Testing')

data_dir_list_Test =np.sort(os.listdir(DATA_PATH_Test))

data_dir_list_Test
img_rows=224

img_cols=224
data_gen = ImageDataGenerator(

    rotation_range=20,

    shear_range=0.5, 

    zoom_range=0.4, 

    rescale=1./255,

    vertical_flip=True, 

    validation_split=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True) 
train_generator = data_gen.flow_from_directory(

        DATA_PATH,

        target_size=(img_rows, img_cols), 

        batch_size=32,

        class_mode='categorical',

        color_mode='rgb', 

        shuffle=True,   

        save_format='png', 

        subset="training")
train_generator.class_indices
test_generator = data_gen.flow_from_directory(

        DATA_PATH_Test,

        target_size=(img_rows, img_cols),

        batch_size=32,

        class_mode='categorical',

        color_mode='rgb', 

        shuffle=True, 

        seed=None,  

        save_format='png',

        subset="validation")
test_generator.class_indices
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(train_generator, epochs=50, validation_data=test_generator,steps_per_epoch=5,validation_steps=2)
fd_model_evaluate = model.evaluate_generator(test_generator, verbose=1)
model.summary()
#model.history.history
#plt.plot(model.history.history["loss"],label ='Traing Loss')

#plt.plot(model.history.history["val_loss"],label ='Validation Loss')

#plt.legend()

#plt.show()

#

#plt.plot(model.history.history["accuracy"],label="Training Accuracy")

#plt.plot(model.history.history["val_accuracy"],label="Validation Accuracy")

#plt.legend()

#plt.show()

#

#(ls,acc)=model.evaluate(x=X_test,y=y_test)

#print(f'MODEL ACCURACY = {acc*100}')
print("Loss: ", fd_model_evaluate[0], "Accuracy: ", fd_model_evaluate[1])
fd_model_predict = model.predict_generator(test_generator, verbose=1)
fd_model_predict.argmax(axis=-1)