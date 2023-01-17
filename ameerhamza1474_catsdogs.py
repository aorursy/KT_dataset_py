# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
import os, shutil

original_dataset_dir = '../input/dogs-vs-cats/train/train'

base_dir = '../input/data'

os.mkdir(base_dir)


train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_cats_dir, fname)

    shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_cats_dir, fname)

    shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_cats_dir, fname)

    shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_dogs_dir, fname)

    shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_dogs_dir, fname)

    shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dst)
print('total training cat images:', len(os.listdir(train_cats_dir)))
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
from keras import optimizers

model.compile(loss='binary_crossentropy',

optimizer=optimizers.Adam(lr=1e-4),

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

validation_dir,target_size=(150, 150),

batch_size=20,

class_mode='binary')
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



for data_batch, labels_batch in train_generator:

    

    imgplot = plt.imshow(data_batch[19])

    plt.show()

    break

history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs=30,

validation_data=validation_generator,

validation_steps=50)
model.load_weights('cats_and_dogs_small_1.h5')
import matplotlib.pyplot as plt

img_path = '../input/dogs-vs-cats/test1/test1/2578.jpg'

from keras.preprocessing import image

import numpy as np

img = image.load_img(img_path, target_size=(150, 150))

img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

print(img_tensor.shape)

plt.imshow(img_tensor[0])
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
first_layer_activation = activations[0]

activations = activation_model.predict(img_tensor)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
layer_names = []

for layer in model.layers[:8]:

    layer_names.append(layer.name)

    images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0,:, :,col * images_per_row + row]

            channel_image -= channel_image.mean()

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size,

            row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

    scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
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
datagen = ImageDataGenerator(

rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest')
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3]

img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i=0

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

optimizer=optimizers.Adam(lr=1e-4),

metrics=['acc'])
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

batch_size=32,

class_mode='binary')

validation_generator = test_datagen.flow_from_directory(

validation_dir,

target_size=(150, 150),

batch_size=32,

class_mode='binary')

history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs=100,

validation_data=validation_generator,

validation_steps=50)
model.save('cats_and_dogs_small_2.h5')
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',

include_top=False,

input_shape=(150, 150, 3))
import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

base_dir = '../input/dogs-vs-cats'

train_dir = os.path.join(base_dir, 'train')

os.mkdir("../input/dogs-vs-cats/validation")

os.mkdir("../input/dogs-vs-cats/test")

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(

    directory,

    target_size=(150, 150),

    batch_size=batch_size,

    class_mode='binary')

    i=0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break

    return features, labels
train_features, train_labels = extract_features(train_dir, 2000)

validation_features, validation_labels = extract_features(validation_dir, 1000)

test_features, test_labels = extract_features(test_dir, 1000)
train2=np.append(train_features, validation_features, axis=0)

trainlab2=np.append(train_features, validation_features, axis=0)
model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),

loss='binary_crossentropy',

metrics=['acc'])

history = model.fit(train2, trainlab2,

epochs=30,

batch_size=20,

validation_data=(validation_features, validation_labels), validation_steps=50)