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
#Input Library pada python

#keras

#numpy

#os

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K



from keras.models import load_model

from keras.preprocessing import image

import numpy as np

from os import listdir

from os.path import isfile, join

#menentukan ukuran gambar ketika ditampilkan

img_width = 150

img_height = 300



#deklarasi folder pada kaggle

train_data_dir = '/kaggle/input/platnomor/images/train'

validation_data_dir = '/kaggle/input/platnomor/images/validation'

train_samples = 120

validation_samples = 30

epochs = 5

batch_size = 20



#check pada tensor flow

if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)

model = Sequential()



#proses pooling gambar pada folder

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

#konversi gambar menjadi 1 dimensi dan menjadikanb dense layer

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))
import keras

from keras import optimizers

model.compile(loss='binary_crossentropy', 

              optimizer=keras.optimizers.Adam(lr=.0001),

              metrics=['accuracy'])
#mengatur ulang gambar setelah pooling untuk ditampilkan

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
#mengatur ulang gambar pada folder test untuk ditampilkan

test_datagen = ImageDataGenerator(rescale=1. / 255)
#membagi folder gambar train menjadi dua kelas

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
#memberi label pada masing - masing class

print(train_generator.class_indices)
#memberikan label pada gambar sesuai class nya

imgs, labels = next(train_generator)
#menampilkan gambar dalam rgb

from skimage import io



def imshow(image_RGB):

  io.imshow(image_RGB)

  io.show()
#menampilkan gambar beserta label nya setelah melalui klasifikasi

import matplotlib.pyplot as plt

%matplotlib inline

image_batch,label_batch = train_generator.next()



print(len(image_batch))

for i in range(0,len(image_batch)):

    image = image_batch[i]

    print(label_batch[i])

    imshow(image)
#membagi folder gambar train menjadi dua kelas

validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
#menentukan akurasi antara data train dengan data validation

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validation_samples // batch_size)
import matplotlib.pyplot as plt

%matplotlib inline



# menampilkan data histori

print(history.history.keys())

# menampilkan hasil akurasi dalam bentuk grafik

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# menampilkan hasil loss dalam bentuk grafik

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#memprediksi hasil klasifikasi gambar pada folder test

test_data_dir = '/kaggle/input/platnomor/images/test/'

onlyfiles = [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f))]

print(onlyfiles)
#hasil klasifikasi image

from keras.preprocessing import image

Foreign_counter = 0 

Indonesia_counter  = 0

for file in onlyfiles:

    img = image.load_img(test_data_dir+file, target_size=(img_width, img_height))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    

    images = np.vstack([x])

    classes = model.predict_classes(images, batch_size=10)

    classes = classes[0][0]

    

    if classes == 0:

        print(file + ": " + 'Foreign')

        Foreign_counter += 1

    else:

        print(file + ": " + 'Indonesia')

        Indonesia_counter += 1

print("Total Foreign :",Foreign_counter)

print("Total Indonesia :",Indonesia_counter)