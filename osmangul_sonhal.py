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
train_dosya="../input/tabakbardakdata/train"

valid_dosya="../input/tabakbardakdata/valid"

test_dosya="../input/tabakbardakdata/test"



bardak_train_dosya="../input/tabakbardakdata/train/bardak"

tabak_train_dosya="../input/tabakbardakdata/train/tabak"



bardak_valid_dosya="../input/tabakbardakdata/valid/bardak"

tabak_valid_dosya="../input/tabakbardakdata/valid/tabak"



bardak_test_dosya="../input/tabakbardakdata/test/bardak"

tabak_test_dosya="../input/tabakbardakdata/test/tabak"
print('Toplam bardak içerikli eğitim verisi', len(os.listdir(bardak_train_dosya)))

print('Toplam bardak içerikli test verisi', len(os.listdir(bardak_test_dosya)))

print('Toplam bardak içerikli valid verisi', len(os.listdir(bardak_valid_dosya)))

print('Toplam tabak içerikli eğitim verisi', len(os.listdir(tabak_train_dosya)))

print('Toplam tabak içerikli test verisi', len(os.listdir(tabak_test_dosya)))

print('Toplam tabak içerikli valid verisi', len(os.listdir(tabak_valid_dosya)))
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



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dosya,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=10,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=9,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=5,

      validation_data=validation_generator,

      validation_steps=4)
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

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
# This is module with image preprocessing utilities

from keras.preprocessing import image



fnames = [os.path.join(bardak_train_dosya, fname) for fname in os.listdir(bardak_train_dosya)]



# We pick one image to "augment"

img_path = fnames[3]



# Read the image and resize it

img = image.load_img(img_path, target_size=(150, 150))



# Convert it to a Numpy array with shape (150, 150, 3)

x = image.img_to_array(img)



# Reshape it to (1, 150, 150, 3)

x = x.reshape((1,) + x.shape)



# The .flow() command below generates batches of randomly transformed images.

# It will loop indefinitely, so we need to `break` the loop at some point!

i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
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



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)





from keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dosya,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=10,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=9,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=4)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
from keras.regularizers import l2

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(512,kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001),activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.3,

      horizontal_flip=True,

      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dosya,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=10,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=9,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=4)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()