# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
        #print("")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data'))
test_def=len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test/def_front'))
test_ok=len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test/ok_front'))
train_ok=len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/ok_front'))
train_def=len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/def_front'))
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['train_ok', 'train_def', 'test_ok', 'test_def']
students = [train_ok,train_def,test_ok,test_def]
ax.bar(langs,students)
plt.show()
train_def/train_ok
class_weight = {0: 1.,
                1:1.3 }
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
from keras.preprocessing.image import ImageDataGenerator

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        '../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train',  # this is the target directory
        target_size=(224, 224),  
        batch_size=batch_size,
        class_mode='binary')  

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')
hist=model.fit_generator(
        train_generator,
        steps_per_epoch=6633 // batch_size,
        epochs=50,class_weight=class_weight,
        validation_data=validation_generator,
        validation_steps=715 // batch_size)
history=hist
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
