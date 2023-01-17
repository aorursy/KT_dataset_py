# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from zipfile import ZipFile 
# importing libraries for Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
test="../input/dogs-cats-images/dog vs cat/dataset/test_set"
train="../input/dogs-cats-images/dog vs cat/dataset/training_set"


train_dog=(train+"/dogs")
train_cat=(train+"/cats")
test_dog=(test+"/dogs")
test_cat=(test+"/cats")
print('number of cats training images - ',len(os.listdir(train_cat)))
print('number of dogs training images - ',len(os.listdir(train_dog)))
print('number of cats testing images - ',len(os.listdir(test_cat)))
print('number of dogs testing images - ',len(os.listdir(test_dog)))
img = load_img(train_dog + "/dog.1004.jpg")
plt.axis("off")
plt.imshow(img)
train_datagen=ImageDataGenerator(rescale = 1.0/255.0,           #resimleri ayrıca scale edelim
                             featurewise_center=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             rotation_range=40,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale= 1./255)


batch_size = 32

train_generator = train_datagen.flow_from_directory(directory = train,        #resmin üretilip depolanacağı yer
                                              target_size = (64,64),        #resimlerimizle aynı shape de olmalı
                                              class_mode = 'binary',          #sadece 2 sınıfımız olduğu için binary
                                              batch_size=batch_size,
                                              shuffle=True)                   #shuffle (karıştırmak) datamızı karıştıralım.
                               

test_generator = test_datagen.flow_from_directory(directory = test,
                                             target_size = (64,64),
                                             batch_size=batch_size,
                                             class_mode = 'binary',
                                             shuffle=True)
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3),input_shape =(64,64,3),padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Conv2D(filters = 128, kernel_size = (3, 3),padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units = 128))
model.add(Activation('relu'))

model.add(Dense(units = 256))
model.add(Activation('relu'))
model.add(Dropout(rate = 0.5))
          
model.add(Dense(units = 2))
model.add(Activation('softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
hist = model.fit_generator(
                        generator=train_generator,
                        steps_per_epoch = 500,
                        epochs = 50,
                        validation_data = test_generator,
                        validation_steps = 400)
print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train loss")
plt.plot(hist.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

plt.plot(hist.history["acc"], label="Train accuracy")
plt.plot(hist.history["val_acc"],label="Validation accuracy")
plt.legend()
plt.show()