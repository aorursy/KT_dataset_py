import numpy as np
import matplotlib.pyplot as plt

def graphing(model):
    acc = model.history['acc']
    val_acc = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']

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
    
def avg_stats(model):
    acc = model.history['acc']
    val_acc = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    
    print('AVG acc: ', np.mean(acc))
    print('AVG val_acc: ', np.mean(val_acc))
    print('AVG loss: ', np.mean(loss))
    print('AVG val_loss: ', np.mean(val_loss))

import os, shutil
import keras
from keras import optimizers
from keras import layers
from keras import models
keras.__version__





original_dataset_dir = '../input/kalem-kumanda/kalem_kumanda'
os.listdir(original_dataset_dir)



train_kalem_dir="../input/kalem-kumanda/kalem_kumanda/train/kalem"
train_kumanda_dir="../input/kalem-kumanda/kalem_kumanda/train/kumanda"
validation_kalem_dir="../input/kalem-kumanda/kalem_kumanda/validation/kalem"
validation_kumanda_dir="../input/kalem-kumanda/kalem_kumanda/validation/kumanda"
test_kalem_dir="../input/kalem-kumanda/kalem_kumanda/test/kalem"
test_kumanda_dir="../input/kalem-kumanda/kalem_kumanda/test/kumanda"
print('Train: kalem fotoğrafı:', len(os.listdir(train_kalem_dir)))
print('Train: kumanda fotoğrafı:', len(os.listdir(train_kumanda_dir)))
print('Validation: kalem fotoğrafı:', len(os.listdir(validation_kalem_dir)))
print('Validation: kumanda fotoğrafı:', len(os.listdir(validation_kumanda_dir)))
print('Test: kalem fotoğrafı:', len(os.listdir(test_kalem_dir)))
print('Test: kumanda fotoğrafı:', len(os.listdir(test_kumanda_dir)))

from keras.preprocessing.image import ImageDataGenerator

train_dir="../input/kalem-kumanda/kalem_kumanda/train"
validation_dir="../input/kalem-kumanda/kalem_kumanda/validation"
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])







history = model.fit_generator(
      train_generator,
      steps_per_epoch=30,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)




graphing(history)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])




history = model.fit_generator(
      train_generator,
      train_generator.n/train_generator.batch_size,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)




avg_stats(history)




graphing(history)




model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])




history = model.fit_generator(
      train_generator,
      steps_per_epoch=30,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50)




avg_stats(history)




graphing(history)


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
from keras.preprocessing import image
import matplotlib.pyplot as plt

fnames = [os.path.join(train_kumanda_dir, fname) for fname in os.listdir(train_kumanda_dir)]

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
from keras.preprocessing import image
import matplotlib.pyplot as plt

fnames = [os.path.join(train_kalem_dir, fname) for fname in os.listdir(train_kalem_dir)]

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
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=30,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


avg_stats(history)




graphing(history)

