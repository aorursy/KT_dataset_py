# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

keras.__version__
import os, shutil

original_dataset_dir = '../input/train'



base_dir = 'canta_vs_telefon'

os.mkdir(base_dir)



train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)



train_canta_dir = os.path.join(train_dir, 'canta')

os.mkdir(train_canta_dir)



train_telefon_dir = os.path.join(train_dir, 'telefon')

os.mkdir(train_telefon_dir)



validation_canta_dir = os.path.join(validation_dir, 'canta')

os.mkdir(validation_canta_dir)



validation_telefon_dir = os.path.join(validation_dir, 'telefon')

os.mkdir(validation_telefon_dir)



fnames = ['canta.{}.jpg'.format(i) for i in range(70)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_canta_dir, fname)

    shutil.copyfile(src, dst)



fnames = ['canta.{}.jpg'.format(i) for i in range(70, 120)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_canta_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['telefon.{}.jpg'.format(i) for i in range(70)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_telefon_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['telefon.{}.jpg'.format(i) for i in range(70, 120)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_telefon_dir, fname)

    shutil.copyfile(src, dst)
import os, shutil

original_dataset_dir = '../input/test'



base_dir = 'canta_vs_telefon_test'

os.mkdir(base_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



fnames = ['{}.jpg'.format(i) for i in range(1,101)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dir, fname)

    shutil.copyfile(src, dst)
print('total training canta images:', len(os.listdir(train_canta_dir)))

print('total training telefon images:', len(os.listdir(train_telefon_dir)))

print('total validation canta images:', len(os.listdir(validation_canta_dir)))

print('total validation telefon images:', len(os.listdir(validation_telefon_dir)))

print('total test images:', len(os.listdir(test_dir)))
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
history = model.fit_generator(

      train_generator,

      steps_per_epoch=7,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=5)
model.save('canta_vs_telefon_1.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary()



from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=7,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=5)
model.save('canta_vs_telefon_2.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=7,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=5)
model.save('canta_vs_telefon_3.h5')

import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()
from keras.layers import Conv2D

from keras import models

from keras.regularizers import l2

from keras import optimizers



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2(0.001), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=7,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=5)
model.save('canta_vs_telefon_4.h5')

import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Loss')

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



from keras.preprocessing import image



fnames = [os.path.join(train_canta_dir, fname) for fname in os.listdir(train_canta_dir)]



img_path = fnames[3]



img = image.load_img(img_path, target_size=(150, 150))



x = image.img_to_array(img)



x = x.reshape((1,) + x.shape)



i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
from keras.layers import Conv2D

from keras import models

from keras.regularizers import l2

from keras import optimizers



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2(0.001), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=7,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=5)
model.save('canta_vs_telefon_5.h5')

import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()