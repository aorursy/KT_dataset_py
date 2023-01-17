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
import random
import zipfile
to_extract = ['train', 'test1']

for file in to_extract:
    with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/"+file+".zip", 'r') as z:
        z.extractall(".")
print(len(os.listdir('train/')))
os.mkdir('train/cats')
os.mkdir('train/dogs')
os.mkdir('test1/cats')
os.mkdir('test1/dogs')
file_list = os.listdir("train/")
for file_name in file_list:
    if(file_name.startswith("cat")):
        shutil.move("train/"+file_name, "train/cats")
    elif(file_name.startswith("dog")):
        shutil.move("train/"+file_name, "train/dogs")
print(len(os.listdir('train/cats/')))
print(len(os.listdir('train/dogs')))
os.mkdir('train1/')
os.mkdir('train1/cats')
os.mkdir('train1/dogs')
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):    
    dataset = []
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData
        if(os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print("skipped" + unitData)
    train_data_length = int(len(dataset) * split_size)
    test_data_length = int(len(dataset) - train_data_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = shuffled_set[0:train_data_length]
    test_set = shuffled_set[-test_data_length:]
    
    for unitData in train_set:
        temp_train_data = SOURCE + unitData
        final_train_data = TRAINING + unitData
        shutil.copyfile(temp_train_data, final_train_data)
    
    for unitData in test_set:
        temp_test_data = SOURCE + unitData
        final_test_data = TESTING + unitData
        shutil.copyfile(temp_train_data, final_test_data)

CAT_SOURCE_DIR = "train/cats/"
TRAINING_CATS_DIR = "train1/cats/"
TESTING_CATS_DIR = "test1/cats/"
DOG_SOURCE_DIR = "train/dogs/"
TRAINING_DOGS_DIR = "train1/dogs/"
TESTING_DOGS_DIR = "test1/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
print(len(os.listdir('train/cats/')))
print(len(os.listdir('train/dogs/')))
print(len(os.listdir('train1/cats/')))
print(len(os.listdir('train1/dogs/')))
print(len(os.listdir('test1/cats/')))
print(len(os.listdir('test1/cats/')))
import tensorflow as tf
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import image
import numpy as np
from PIL import Image
model = Sequential([
    Convolution2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),
    
    Convolution2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
class cust_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > .98 and logs.get('val_acc') > .98):
            print("Training stopped, reached max accuracy")
            self.model.stop_training=True
callback = cust_callback()
train_dir = 'train1/'
test_dir = 'test1/'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(300, 300), 
                                                    batch_size = 250,
                                                    class_mode = 'binary'
                                                   )
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                    target_size=(300, 300), 
                                                    batch_size = 250,
                                                    class_mode = 'binary'
                                                   )
history = model.fit_generator(test_generator, steps_per_epoch=45, epochs=10, validation_data=test_generator,callbacks=[callback], verbose=1)
model.save_weights('cats_dogs_model.h5')
model.save('c&dmodel.h5')
 
