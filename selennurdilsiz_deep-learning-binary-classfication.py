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
!pwd
os.chdir('..')

os.listdir()
import os, shutil



#Eğitilecek veri seti

original_dataset_dir = 'input/train'

#Modelleri kaydetmek için

base_dir = 'pens_and_books_small'

os.mkdir(base_dir)



#Veri train,test ve validation şeklinde bölündü.

train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



train_pens_dir = os.path.join(train_dir, 'pens')

os.mkdir(train_pens_dir)





train_books_dir = os.path.join(train_dir, 'books')

os.mkdir(train_books_dir)





validation_pens_dir = os.path.join(validation_dir, 'pens')

os.mkdir(validation_pens_dir)





validation_books_dir = os.path.join(validation_dir, 'books')

os.mkdir(validation_books_dir)





test_pens_dir = os.path.join(test_dir, 'pens')

os.mkdir(test_pens_dir)





test_books_dir = os.path.join(test_dir, 'books')

os.mkdir(test_books_dir)



# 70 kalem resmi train_pens_dir kopyalandı.

fnames = ['kalem.{}.jpg'.format(i) for i in range(70)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_pens_dir, fname)

    shutil.copyfile(src, dst)



# 30 kalem resmi validation_pens_dir kopyalandı.

fnames = ['kalem.{}.jpg'.format(i) for i in range(70, 100)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_pens_dir, fname)

    shutil.copyfile(src, dst)

    

# 50 kalem resmi test_pens_dir kopyalandı.

fnames = ['kalem.{}.jpg'.format(i) for i in range(0, 50)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_pens_dir, fname)

    shutil.copyfile(src, dst)

    

# # 70 kitap resmi train_books_dir kopyalandı.

fnames = ['kitap.{}.jpg'.format(i) for i in range(70)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_books_dir, fname)

    shutil.copyfile(src, dst)

    

# # 30 kitap resmi validation_books_dir kopyalandı.

fnames = ['kitap.{}.jpg'.format(i) for i in range(70, 100)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_books_dir, fname)

    shutil.copyfile(src, dst)

    

# # 50 kitap resmi test_books_dir kopyalandı.

fnames = ['kitap.{}.jpg'.format(i) for i in range(0, 50)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_books_dir, fname)

    shutil.copyfile(src, dst)
print('Total training pen images:', len(os.listdir(train_pens_dir)))
print('Total training book images:', len(os.listdir(train_books_dir)))
print('Total validation pen images: ', len(os.listdir(validation_pens_dir)))
print('Total validation book images: ', len(os.listdir(validation_books_dir)))
print('Total test pen images:', len(os.listdir(test_pens_dir)))
print('Total test book images:', len(os.listdir(test_books_dir)))

from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))

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

                    target_size = (300, 300),

                    batch_size = 10,

                    class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

                        validation_dir,

                        target_size = (300, 300),

                        batch_size = 10,

                        class_mode = 'binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
history = model.fit_generator(

                train_generator,

                steps_per_epoch = 6,

                epochs = 6,

                validation_data = validation_generator,

                validation_steps = 3)
model.save('pens_and_books_model_v1.h5')
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

from keras import regularizers

model = models.Sequential()



model.add(layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(

                    train_dir,

                    target_size = (300, 300),

                    batch_size = 10,

                    class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

                        validation_dir,

                        target_size = (300, 300),

                        batch_size = 10,

                        class_mode = 'binary')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=20,

      epochs=15,

      validation_data=validation_generator,

      validation_steps=3)
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



model.add(layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

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

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(

                    train_dir,

                    target_size = (300, 300),

                    batch_size = 10,

                    class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

                        validation_dir,

                        target_size = (300, 300),

                        batch_size = 10,

                        class_mode = 'binary')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=20,

      epochs=15,

      validation_data=validation_generator,

      validation_steps=3)
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
from keras import regularizers

from keras.regularizers import l2

model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3),kernel_regularizer=regularizers.l2(0.001),

                       activation = 'relu', input_shape = (300, 300, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(0.001),activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3),kernel_regularizer=regularizers.l2(0.001),activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3),kernel_regularizer=regularizers.l2(0.001),activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(

                    train_dir,

                    target_size = (300, 300),

                    batch_size = 10,

                    class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

                        validation_dir,

                        target_size = (300, 300),

                        batch_size = 10,

                        class_mode = 'binary')



history = model.fit_generator(

                train_generator,

                steps_per_epoch = 20,

                epochs = 15,

                validation_data = validation_generator,

                validation_steps = 3)
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
import matplotlib.pyplot as plt

from keras.preprocessing import image



fnames = [os.path.join(train_pens_dir, fname) for fname in os.listdir(train_pens_dir)]



img_path = fnames[50]



img = image.load_img(img_path, target_size = (300, 300))



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
from keras import regularizers

model = models.Sequential()



model.add(layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

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

        target_size=(300, 300),

        batch_size=10,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(300, 300),

        batch_size=10,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=20,

      epochs=15,

      validation_data=validation_generator,

      validation_steps=3)
model.save('pens_and_books_model_v2.h5')
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