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
!pwd

os.chdir('..')
os.listdir()
import os, shutil

original_dataset_dir = 'input/kus-balik-dataset/TrainDataset/TrainDataset'

base_dir = 'kus_ve_balik_sinifi'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_kuslar_dir = os.path.join(train_dir, 'kuslar')
os.mkdir(train_kuslar_dir)

train_baliklar_dir = os.path.join(train_dir, 'baliklar')
os.mkdir(train_baliklar_dir)

validation_kuslar_dir = os.path.join(validation_dir, 'kuslar')
os.mkdir(validation_kuslar_dir)

validation_baliklar_dir = os.path.join(validation_dir, 'baliklar')
os.mkdir(validation_baliklar_dir)

test_kuslar_dir = os.path.join(test_dir, 'kuslar')
os.mkdir(test_kuslar_dir)

test_baliklar_dir = os.path.join(test_dir, 'baliklar')
os.mkdir(test_baliklar_dir)

fnames = ['Kus.{}.jpg'.format(i) for i in range(100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_kuslar_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['Kus.{}.jpg'.format(i) for i in range(100, 125)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,  fname)
    dst = os.path.join(validation_kuslar_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['Kus.{}.jpg'.format(i) for i in range(125, 150)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_kuslar_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['Balik.{}.jpg'.format(i) for i in range(100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_baliklar_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['Balik.{}.jpg'.format(i) for i in range(100, 125)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_baliklar_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['Balik.{}.jpg'.format(i) for i in range(125, 150)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_baliklar_dir, fname)
    shutil.copyfile(src, dst)
print('Toplam kus train verisi:', len(os.listdir(train_kuslar_dir)))
print('Toplam balik train verisi:', len(os.listdir(train_baliklar_dir)))
print('Toplam kus validation verisi: ', len(os.listdir(validation_kuslar_dir)))
print('Toplam balik validation verisi: ', len(os.listdir(validation_baliklar_dir)))
print('Toplam kus test verisi:', len(os.listdir(test_kuslar_dir)))
print('Toplam balik test verisi:', len(os.listdir(test_baliklar_dir)))
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu',
                       input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()
from keras import optimizers

model.compile(loss = 'binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics = ['acc'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (150, 150),
                    batch_size = 20,
                    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
                        validation_dir,
                        target_size = (150, 150),
                        batch_size = 20,
                        class_mode = 'binary')
for data_batch, labels_batch in train_generator:
    print('veri sekli:', data_batch.shape)
    print('etiket sekli:', labels_batch.shape)
    break
history = model.fit_generator(
                train_generator,
                steps_per_epoch = 100,
                epochs = 20,
                validation_data = validation_generator,
                validation_steps = 50)
model.save('kus_ve_balik_sinifi_1.h5')
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training ve Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training ve Validation Loss')
plt.legend()

plt.show()
datagen = ImageDataGenerator(
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
import matplotlib.pyplot as plt
from keras.preprocessing import image

fnames = [os.path.join(train_kuslar_dir, fname) for fname in os.listdir(train_kuslar_dir)]

img_path = fnames[20]

img = image.load_img(img_path, target_size = (150, 150))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1):
    plt.figure()
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                                train_dir,
                                target_size = (150, 150),
                                batch_size = 32,
                                class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                                validation_dir,
                                target_size = (150, 150),
                                batch_size = 32,
                                class_mode = 'binary')

history = model.fit_generator(
                            train_generator,
                            steps_per_epoch = 100,
                            epochs = 25,
                            validation_data = validation_generator,
                            validation_steps = 50)
model.save('kus_ve_balik_sinifi_2.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training ve Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training ve Validation Loss')
plt.legend()

plt.show()
import os
print(os.listdir('input/vgg16'))
from keras.applications import VGG16

conv_base = VGG16(weights='input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 include_top = False,
                 input_shape = (150, 150, 3))
conv_base.summary()
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
model.compile(loss = 'binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-5),
             metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=25,
    validation_data = validation_generator,
    validation_steps=50)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training ve Validation Accuracy')

plt.figure()
plt.plot(epochs, loss, 'bo', label ='Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training ve validation loss')
plt.legend()

plt.show()
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)