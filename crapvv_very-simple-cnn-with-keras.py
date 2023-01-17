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
import shutil
def prepare_data(clas):
    base_dir = '/'
    validation_dir = '/validation9'
    os.mkdir(validation_dir)
    test_dir = '/kaggle/input/fruits/fruits-360/Test/'
    for cl in clas:
        cl_dir = os.path.join(test_dir, cl)
        train_names = [image for image in os.listdir(cl_dir)]
        os.mkdir(os.path.join(validation_dir, cl))
        cl_vl = os.path.join(validation_dir, cl)
        for i in range(len(train_names)//2):
            src = os.path.join(cl_dir, train_names[i])
            dst = os.path.join(cl_vl, train_names[i])
            shutil.copyfile(src, dst)
    
    new_test = '/test7'
    os.mkdir(new_test)
    for cl in clas:
        cl_dir = os.path.join(test_dir, cl)
        train_names = [image for image in os.listdir(cl_dir)]
        os.mkdir(os.path.join(new_test, cl))
        cl_t = os.path.join(new_test, cl)
        for i in range(len(train_names)//2, len(train_names)):
            src = os.path.join(cl_dir, train_names[i])
            dst = os.path.join(cl_t, train_names[i])
            shutil.copyfile(src, dst)
    
clas = [file for file in os.listdir('/kaggle/input/fruits/fruits-360/Training/')]
prepare_data(clas)
print('Total Validation images : ', sum(len(os.listdir(f'/test7/{file}')) for file in clas))
print('Total Test images : ', sum(len(os.listdir(f'/validation9/{file}')) for file in clas))
from keras.preprocessing.image import ImageDataGenerator

train_dir = '/kaggle/input/fruits/fruits-360/Training/'
validation_dir = '/validation9'
test_dir = '/test7'

data_gen = ImageDataGenerator(rescale=1./255)
data_gen1 = ImageDataGenerator(rescale=1./255)
data_gen2 = ImageDataGenerator(rescale=1./255)

train_gen = data_gen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    class_mode='categorical')

validation_gen = data_gen1.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    class_mode='categorical'
)

test_gen = data_gen2.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    class_mode='categorical'
) 
import tensorflow as tf
from keras import models
from keras import models, layers
call_backs = [
        tf.keras.callbacks.EarlyStopping(patience=4)
 
]
model = models.Sequential()

model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(276, activation='relu'))
model.add(layers.Dense(131, activation='softmax'))

from keras import models, layers
from keras import optimizers
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,  validation_data=validation_gen, epochs=1, callbacks=call_backs)
model.evaluate(test_gen)
