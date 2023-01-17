import numpy as np 

import pandas as pd 

from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from keras.regularizers import l2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
train="../input/bookpen/train"

validation="../input/bookpen/validation"

test="../input/bookpen/test"



train_books="../input/bookpen/train/books"

train_pens="../input/bookpen/train/pens"



validation_books="../input/bookpen/validation/books"

validation_pens="../input/bookpen/validation/pens"



test_books="../input/bookpen/test/books"

test_pens="../input/bookpen/test/pens"
print('total training book images:', len(os.listdir(train_books)))

print('total training pen images:', len(os.listdir(train_pens)))



print('total validation book images:', len(os.listdir(validation_books)))

print('total validation pen images:', len(os.listdir(validation_pens)))



print('total test book images:', len(os.listdir(test_books)))

print('total test pen images:', len(os.listdir(test_pens)))
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
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
# Resimler 1./255 ile yeniden ölçeklendirildi.

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dizin

        train,

        # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=25,

        # binary_crossentropy kullandığımız için binary label'lara ihtiyacımız var.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        # Hedef dizin

        validation,

        # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=15,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=15,

      validation_data=validation_generator,

      validation_steps=10)
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



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
# Tüm resimler 1./255 ile yeniden ölçeklendirildi.

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dizin

        train,

         # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=15,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation,

        target_size=(150, 150),

        batch_size=10,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=15,

      validation_data=validation_generator,

      validation_steps=10)
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



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
# Resimler 1./255 ile yeniden ölçeklendirildi.

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dizin

        train,

        # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=15,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation,

        target_size=(150, 150),

        batch_size=10,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=20,

      epochs=6,

      validation_data=validation_generator,

      validation_steps=5)
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
# Tüm resimler 1./255 ile yeniden ölçeklendirildi.

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dizin

        train,

        # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=15,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation,

        target_size=(150, 150),

        batch_size=10,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=20,

      epochs=6,

      validation_data=validation_generator,

      validation_steps=5)
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

model.add(layers.Dense(512, kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
# Tüm resimler 1./255 ile yeniden ölçeklendirildi.

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dizin

        train,

        # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=15,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation,

        target_size=(150, 150),

        batch_size=10,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=20,

      epochs=6,

      validation_data=validation_generator,

      validation_steps=5)
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
fnames = [os.path.join(train_books, fname) for fname in os.listdir(train_books)]



# Augmentation için bir resim seçildi.

img_path = fnames[3]



# Seçilen resim yeniden boyutlandırıldı.

img = image.load_img(img_path, target_size=(150, 150))



# Resmin Numpy dizisine dönüşütürülmesi. (150, 150, 3)

x = image.img_to_array(img)



# Resmin yeniden boyutlandırılması. (1, 150, 150, 3)

x = x.reshape((1,) + x.shape)



#Boyutlandırılmış yeni resmin gösterilmesi.

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

model.add(layers.Dense(512, kernel_regularizer=l2(0.0001), activation='relu'))

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

    horizontal_flip=True)



# Validation verileri augmente edilmemeli!!.

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dizin

        train,

        # Tüm resimler 150x150 olarak yeniden boyutlandırıldı.

        target_size=(150, 150),

        batch_size=15,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation,

        target_size=(150, 150),

        batch_size=10,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=50,

      epochs=6,

      validation_data=validation_generator,

      validation_steps=5)
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