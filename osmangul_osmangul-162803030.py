import numpy as np 

import pandas as pd





import os

print(os.listdir("../input"))

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



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyanın seçilmesi

        train_dosya,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

         # Hedef dosyanın seçilmesi

        valid_dosya,

        #Resimlerin yeniden boyutlandırılması

        target_size=(150, 150),

        #Parça parça alınacak veri sayısının büyüklüğü    

        batch_size=10,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('verilerimizin şekilsel  gösterimi:', data_batch.shape)

    print('etiketlerimizin sayısı:', labels_batch.shape)

    break
history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=8,

      validation_data=validation_generator,

      validation_steps=10)


#Grafiği ekrana yazdırdık 

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

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyanın seçilmesi

        train_dosya,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

         # Hedef dosyanın seçilmesi

        valid_dosya,

        #Resimlerin yeniden boyutlandırılması

        target_size=(150, 150),

        #Parça parça alınacak veri sayısının büyüklüğü    

        batch_size=10,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=4,

      validation_data=validation_generator,

      validation_steps=10)
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
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_dosya,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=8,

      validation_data=validation_generator,

      validation_steps=10)
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
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(8, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_dosya,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=8,

      validation_data=validation_generator,

      validation_steps=10)
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
from keras.regularizers import l2

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_dosya,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=8,

      validation_data=validation_generator,

      validation_steps=10)
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
#DATA AUGMENTATION

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



fnames = [os.path.join(bardak_train_dosya, fname) for fname in os.listdir(bardak_train_dosya)]



# Datasetimiz içinden bir resmi alıyoruz.

img_path = fnames[5]



# Aldığımız resmin boyutunu yeniden boyutlandırıyoruz

img = image.load_img(img_path, target_size=(150, 150))



# aldığımız verileri dizi altında gösteriyoruz

x = image.img_to_array(img)



# diziyi yeniden boyutlandırıyoruz

x = x.reshape((1,) + x.shape)



i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
from keras.regularizers import l2

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3),kernel_regularizer=l2(0.01),  activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_dosya,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_dosya,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=14,

      epochs=8,

      validation_data=validation_generator,

      validation_steps=10)
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