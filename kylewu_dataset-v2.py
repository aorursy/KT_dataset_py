# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import random# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import cv2
from tqdm.notebook import tqdm 
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
millionFake_Dir_1 = '/kaggle/input/1-million-fake-faces/1m_faces_00_01_02_03/1m_faces_00_01_02_03/1m_faces_02'
millionFake_Dir_2 = '/kaggle/input/1-million-fake-faces/1m_faces_00_01_02_03/1m_faces_00_01_02_03/1m_faces_03'
celebA_Dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
deepppFake_Dir = '/kaggle/input/deepppp/dataset/fake'
deepppReal_Dir = '/kaggle/input/deepppp/dataset/real'
train_fake_dir = '/kaggle/tmp/train/fake'
os.makedirs(train_fake_dir)
val_fake_dir = '/kaggle/tmp/val/fake'
os.makedirs(val_fake_dir)
test_fake_dir = '/kaggle/tmp/test/fake'
os.makedirs(test_fake_dir)
train_real_dir = '/kaggle/tmp/train/real'
os.makedirs(train_real_dir)
val_real_dir = '/kaggle/tmp/val/real'
os.makedirs(val_real_dir)
test_real_dir = '/kaggle/tmp/test/real'
os.makedirs(test_real_dir)
deepppp_fake_paths = []
for directory, _, filenames in tqdm(os.walk(deepppFake_Dir)):
    for filename in filenames:
        file_path = os.path.join(directory,filename)
        deepppp_fake_paths.append(file_path)
deepppp_real_paths = []
for directory, _, filenames in tqdm(os.walk(deepppReal_Dir)):
    for filename in filenames:
        file_path = os.path.join(directory,filename)
        deepppp_real_paths.append(file_path)
print(len(deepppp_fake_paths))
print(len(deepppp_real_paths))
deepppp_fake_paths = random.sample(deepppp_fake_paths,10000)
deepppp_real_paths = random.sample(deepppp_real_paths,10000)
celeb_paths = []
for directory, _, filenames in tqdm(os.walk(celebA_Dir)):
    for filename in filenames:
        celeb_path = os.path.join(directory,filename)
        celeb_paths.append(celeb_path)
print(len(celeb_paths))
celeb_paths = random.sample(celeb_paths,20000)
million_paths = []
for directory, _, filenames in tqdm(os.walk(millionFake_Dir_1)):
    for filename in filenames:
        million_path = os.path.join(directory,filename)
        million_paths.append(million_path)
for directory, _, filenames in tqdm(os.walk(millionFake_Dir_2)):
    for filename in filenames:
        million_path = os.path.join(directory,filename)
        million_paths.append(million_path)
print((len(million_paths)))
fake_paths = []
fake_paths.extend(million_paths)
fake_paths.extend(deepppp_fake_paths)
print(len(fake_paths))
real_paths = []
real_paths.extend(celeb_paths)
real_paths.extend(deepppp_real_paths)
print(len(fake_paths))
random.shuffle(fake_paths)
random.shuffle(real_paths)
indx_list = [n for n in range(30000)]
random.shuffle(indx_list)
train_idx = indx_list[:24000]
val_idx = indx_list[24000:27000]
test_idx = indx_list[27000:30000]
train_fake_paths = [fake_paths[i] for i in train_idx]
val_fake_paths = [fake_paths[i] for i in val_idx]
test_fake_paths = [fake_paths[i] for i in test_idx]
train_real_paths = [real_paths[i] for i in train_idx]
val_real_paths = [real_paths[i] for i in val_idx]
test_real_paths = [real_paths[i] for i in test_idx]
def move_resize_images(save_dir,path_list):
    i = 0
    for path in tqdm(path_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        filename = str(i) + '.jpg'
        cv2.imwrite(os.path.join(save_dir, filename), img)
        i += 1
move_resize_images(train_fake_dir,train_fake_paths)
print('1')
move_resize_images(val_fake_dir,val_fake_paths)
print('2')
move_resize_images(test_fake_dir,test_fake_paths)
print('3')
move_resize_images(train_real_dir,train_real_paths)
print('4')
move_resize_images(val_real_dir,val_real_paths)
print('5')
move_resize_images(test_real_dir,test_real_paths)
print('6')
train_dir = '/kaggle/tmp/train'
val_dir = '/kaggle/tmp/val'
test_dir = '/kaggle/tmp/test'
!apt install tree 
!tree --filelimit 3 /kaggle/tmp
!du -chs /kaggle/tmp
train_datagen = ImageDataGenerator(rescale=1./255,
                            rotation_range=30
                            )
val_datagen = ImageDataGenerator(rescale=1./255,
                            )

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=48,
    class_mode='binary',
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=6,
    class_mode='binary',
)
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf
weights_path = '/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
conv_base = VGG16(weights=weights_path,include_top=False, input_shape=(224, 224, 3))
for layer in conv_base.layers[:-8]:
    layer.trainable = False
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(lr=0.0002),
    metrics=['accuracy']
)
checkpointpath='/kaggle/working/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpointpath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,mode='auto',
                            period=1)
model_history = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=train_gen.samples/train_gen.batch_size,
    validation_steps=val_gen.samples/val_gen.batch_size,
    callbacks=[checkpoint],
    epochs=100,
    verbose=1
)
