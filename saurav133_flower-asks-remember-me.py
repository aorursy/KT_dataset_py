# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname))

# Any results you write to the current directory are saved as output.
from os.path import join

image_dir = '../input/flowers-recognition/flowers/'
img_paths = []
img_labels = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        img_paths.append((os.path.join(dirname, filename)))
        img_labels.append(dirname.split('/')[-1])
img_paths[:10]
img_labels[:10]
len(img_paths), len(img_labels)
import matplotlib.pyplot as plt
import random
import cv2

fig, ax = plt.subplots(4,3)
fig.set_size_inches(15, 15)
for i in range(4):
    for j in range(3):
        rand = random.randint(0, len(img_paths))
        image = cv2.imread(img_paths[rand])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[i,j].imshow(image)
        ax[i,j].set_title('Flower_name: '+img_labels[rand])
        
plt.tight_layout()
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(img_paths, img_labels, test_size=0.15, random_state=101)
len(train_paths), len(val_paths), len(train_labels), len(val_labels)
import os
import shutil

t_path = '/kaggle/working/train'
v_path = '/kaggle/working/val'

if not os.path.exists(t_path):
    os.mkdir(t_path)
else:
    shutil.rmtree(t_path)
    os.mkdir(t_path)
    
if not os.path.exists(v_path):
    os.mkdir(v_path)
else:
    shutil.rmtree(v_path)
    os.mkdir(v_path)
    
    
train_set = [(file, label) for file, label in zip(train_paths, train_labels)]
for file, label in train_set:
    path = t_path + '/' + label
    if not os.path.exists(path):
        os.mkdir(path)
    shutil.copyfile(file, os.path.join(path, str(random.randint(0, 9999))+file.split('/')[-1]))


val_set = [(file, label) for file, label in zip(val_paths, val_labels)]
for file, label in val_set:
    path = v_path + '/' + label
    if not os.path.exists(path):
        os.mkdir(path)
    shutil.copyfile(file, os.path.join(path, str(random.randint(0, 9999))+file.split('/')[-1]))
    
    
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

num_classes = 5

res_net_model = Sequential()
res_net_model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
res_net_model.add(Dense(128, activation='relu'))
res_net_model.add(Dropout(0.5))
res_net_model.add(Dense(256, activation='relu'))
res_net_model.add(Dense(num_classes, activation='softmax'))

res_net_model.layers[0].trainable = False
res_net_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 200
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
batch_size = 64
train_generator = data_generator.flow_from_directory(
        '/kaggle/working/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/kaggle/working/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

count = sum([len(files) for r, d, files in os.walk("/kaggle/working/train")])
print(count)
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_lr=0.0001)

res_net_model.fit_generator(train_generator,
                            steps_per_epoch=int(count/batch_size) + 1,
                            epochs=10,
                            validation_data=validation_generator,
                            validation_steps=1,
                            callbacks=[reduce_lr])
history = res_net_model.history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#model from scratch
my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding = 'Same', input_shape=(image_size, image_size, 3), activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Conv2D(128, kernel_size=(3, 3), strides=2, padding = 'Same', activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.2))
my_model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding = 'Same', activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding = 'Same', activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Conv2D(128, kernel_size=(3, 3), strides=2, padding = 'Same', activation='relu'))
my_model.add(Flatten())
my_model.add(Dense(128, activation='relu'))
my_model.add(Dense(256, activation='relu'))
my_model.add(Dropout(0.2))
my_model.add(Dense(256, activation='relu'))
my_model.add(Dense(num_classes, activation='softmax'))

my_model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

my_model.fit_generator(train_generator,
                       steps_per_epoch=int(count/batch_size) + 1,
                       epochs=20,
                       validation_data=validation_generator,
                       validation_steps=1,
                       callbacks=[reduce_lr])
history = my_model.history

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
