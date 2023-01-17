# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#check if my data were correctly uploaded

print(os.listdir('/kaggle/input/cats_and_dogs_small/cats_and_dogs_small/'))
print('Total training cats images', len(os.listdir('/kaggle/input/cats_and_dogs_small/cats_and_dogs_small/train/cats/')))
#instantiating a small convnet for dogs vs cats classification

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='relu'))
model.summary()
#configuring the model for training

from keras import optimizers



model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
#we have to preprocess our data

from keras.preprocessing.image import ImageDataGenerator

train_dir = '/kaggle/input/cats_and_dogs_small/cats_and_dogs_small/train'

validation_dir = '/kaggle/input/cats_and_dogs_small/cats_and_dogs_small/validation'

train_datagen = ImageDataGenerator(rescale=1.0/255) #we rescale by a factor of 1/255

test_datagen = ImageDataGenerator(rescale=1.0/255)



train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=20,

                                                   class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150),

                                                       batch_size=20, class_mode='binary')
#fitting a model using a batch generator

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, 

                             validation_data=validation_generator, validation_steps=50)
#saving the model

model.save('cats_and_dogs_small_1.h5')
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
datagen = ImageDataGenerator(rotation_range=40,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True,

                            fill_mode='nearest')
train_cats_dir = '/kaggle/input/cats_and_dogs_small/cats_and_dogs_small/train/cats/'

from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3]

img = image.load_img(img_path, target_size=(150,150))

x = image.img_to_array(img) #converts the image to a numpy array with shape (150,150,3)

x = x.reshape((1,) + x.shape) #je mets l'image dans le tirroir



i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    if i%4 == 0:

        break;

plt.show()
#a new convnet with dropout

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
#training the convnet using data-augmentation generators

train_datagen = ImageDataGenerator(

rescale=1./255,

rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,)



test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,

                                                   target_size=(150,150),

                                                   batch_size=32,

                                                   class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,

                                                       target_size=(150,150),

                                                       batch_size=32,

                                                       class_mode='binary')

history = model.fit_generator(train_generator,

                             steps_per_epoch=100,

                             epochs=100,

                             validation_data=validation_generator,

                             validation_steps=50)
model.save('cats_and_dogs_small_2.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.summary()
#extracting features using the pretrained convolutional base

from keras.preprocessing.image import ImageDataGenerator

import os

import numpy as np



base_dir = '/kaggle/input/cats_and_dogs_small/cats_and_dogs_small/'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4,4,512))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(directory,

                                           target_size=(150,150),

                                           batch_size=batch_size,

                                           class_mode='binary')

    i=0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size: (i+1)*batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i+=1

        if i * batch_size >= sample_count:

            break;

    return features, labels
train_features, train_labels = extract_features(train_dir, 2000)

validation_features, validation_labels = extract_features(validation_dir, 1000)

test_features, test_labels = extract_features(test_dir, 1000)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))

validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))

test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
#defining and training the densely connected classifier

from keras import models

from keras import layers

from keras import optimizers



model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_dim=(4 * 4 * 512)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_features,

                   train_labels,

                   epochs=30,

                   batch_size=20,

                   validation_data=(validation_features, validation_labels))
#plotting the results

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
#feature extraction with data augmentation

#adding a densely classifier on top of the convolutional base

network = models.Sequential()

network.add(conv_base)

network.add(layers.Flatten())

network.add(layers.Dense(256, activation='relu'))

network.add(layers.Dense(1, activation='sigmoid'))

network.summary()
conv_base.trainable= False
network.summary()
#Training the model end to end with a frozen convolutional base

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,

                                                   target_size=(150,150),

                                                   batch_size=20,

                                                   class_mode='binary')

validation_generator = train_datagen.flow_from_directory(validation_dir,

                                                        target_size=(150,150),

                                                        batch_size=20,

                                                        class_mode='binary')

network.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = network.fit_generator(train_generator,

                               steps_per_epoch=100,

                               epochs=30,

                               validation_data=validation_generator,

                               validation_steps=50)

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
conv_base.summary()
#evaluate the model

test_generator = datagen.flow_from_directory(test_dir,

                                            target_size=(150,150),

                                            batch_size=20,

                                            class_mode='binary')

test_loss, test_acc = network.evaluate_generator(test_generator, steps=50)

print('test_acc', test_acc)