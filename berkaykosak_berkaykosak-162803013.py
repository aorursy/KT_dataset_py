

import os, shutil

import keras

keras.__version__

import numpy as np 

import pandas as pd



print(os.listdir("../input"))

#kütüphanelerimizi import ettik.
train_klasoru="../input/bilgitabanli/train"

valid_klasoru="../input/bilgitabanli/valid"

test_klasoru="../input/bilgitabanli/test"



bozukPara_train_klasoru="../input/bilgitabanli/train/bozukPara"

kagitPara_train_klasoru="../input/bilgitabanli/train/kagitPara"



bozukPara_valid_klasoru="../input/bilgitabanli/valid/bozukPara"

kagitPara_valid_klasoru="../input/bilgitabanli/valid/kagitPara"



bozukPara_test_klasoru="../input/bilgitabanli/test/bozukPara"

kagitPara_test_klasoru="../input/bilgitabanli/test/kagitPara"



#train,test,validation olarak ayırdığımız klasörlerimizin dosya uzanlatılarını yazarak atadığımız değişkenlere bağladık
bozukPara_train_klasoru = "../input/bilgitabanli/train/bozukPara"

#verileri çekip çekmediğini test edeceğimiz dosya uzantısını girdik .
deneme = os.listdir(bozukPara_train_klasoru)

deneme

#deneme fonk. ile train klasöründeki bozuk para fotoğralarını listeledik.
print('train içindeki toplam bozuk para sayısı:', len(os.listdir(bozukPara_train_klasoru )))

#klasörümüze bulunan toplam sayıyı görüntüledik kontrollerimi yaptık.
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

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()



#derste öğrendiğimiz verilerle çalıştık 150x150 boyutunda shape ettik. aktivasyon fonksiyonu olarak relu ve sigmoid kullandık.

#conv2D değerleri için default olan yüksek değerleri kullandık.Diğer modeller de bunlarla oynayarak değişimi takip edicez.
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break

    #
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

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

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break

    
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=5,

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
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))



model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break

    
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=7,

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
from keras import layers

from keras import models



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



model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break

    
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=7,

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

model.add(layers.Conv2D(32, (3, 3),kernel_regularizer=l2(0.01),  activation='relu',

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
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

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



fnames = [os.path.join(bozukPara_train_klasoru, fname) for fname in os.listdir(bozukPara_train_klasoru)]



# herhangi bir resmi örnek olarak seçiyoruz

img_path = fnames[7]



# resimlerin boyutunu ayarla

img = image.load_img(img_path, target_size=(150, 150))



# resim boyutlandırması ve renk (150, 150, 3)

x = image.img_to_array(img)

#aldığımız verileri dizide gösteriyoruz



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

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=7,

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

      zoom_range=0.1,

      horizontal_flip=True,

      fill_mode='nearest')
from keras.preprocessing import image

import matplotlib.pyplot as plt



fnames = [os.path.join(kagitPara_train_klasoru, fname) for fname in os.listdir(kagitPara_train_klasoru)]



# herhangi bir resmi örnek olarak seçiyoruz

img_path = fnames[5]



# resimlerin boyutunu ayarla

img = image.load_img(img_path, target_size=(150, 150))



# resim boyutlandırması ve renk (150, 150, 3)

x = image.img_to_array(img)

#aldığımız verileri dizide gösteriyoruz



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

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=7,

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



fnames = [os.path.join(kagitPara_train_klasoru, fname) for fname in os.listdir(kagitPara_train_klasoru)]



# herhangi bir resmi örnek olarak seçiyoruz

img_path = fnames[5]



# resimlerin boyutunu ayarla

img = image.load_img(img_path, target_size=(150, 150))



# resim boyutlandırması ve renk (150, 150, 3)

x = image.img_to_array(img)

#aldığımız verileri dizide gösteriyoruz



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

                        input_shape=(150, 150,3)))

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
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      train_generator.n/train_generator.batch_size,

      epochs=7,

      validation_data=validation_generator,

      validation_steps=50)
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

      zoom_range=0.1,

      horizontal_flip=True,

      fill_mode='nearest')
from keras.preprocessing import image

import matplotlib.pyplot as plt



fnames = [os.path.join(bozukPara_train_klasoru, fname) for fname in os.listdir(bozukPara_train_klasoru)]



# herhangi bir resmi örnek olarak seçiyoruz

img_path = fnames[5]



# resimlerin boyutunu ayarla

img = image.load_img(img_path, target_size=(150, 150))



# resim boyutlandırması ve renk (150, 150, 3)

x = image.img_to_array(img)

#aldığımız verileri dizide gösteriyoruz



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

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(8, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))##???? silmeyi dene bide



model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(16, kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])



#loss olarak binary-crossentropy kullandık 2 li seçim yapacağımızdan

#optimizer olaraksa RMSprop kullandık

#metrik oalrak accuracy değerini kullanıcaz.
from keras.preprocessing.image import ImageDataGenerator



#tüm resimleri 255 rescale boyutuna getirdik

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # Hedef dosyamız

        train_klasoru,

        # tüm resimler 150x150 formatına dönüştürmüştük yukarı.

        target_size=(150, 150),

        batch_size=20,

        # 2 li seçim classifaction yapacağımızdan binary class modunu kullandık.

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_klasoru,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=7,

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