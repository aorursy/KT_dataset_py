import os

from os import listdir, makedirs

from os.path import join, exists, expanduser



from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dense, MaxPooling2D, BatchNormalization, Conv2D

from keras.layers import Activation, Dropout, Flatten, Dense

import matplotlib.pyplot as plt
train_data_dir = '../input/fruits-360_dataset/fruits-360/Training'

test_data_dir = '../input/fruits-360_dataset/fruits-360/Test'

train_samples = 59328

test_samples = 20232

img_height = 100

img_width = 100

batch_size = 40
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(train_data_dir,

                                                    target_size=(img_height, img_width),

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    subset='training')



val_generator = train_datagen.flow_from_directory(train_data_dir,

                                                  target_size=(img_height, img_width),

                                                  batch_size=batch_size,

                                                  class_mode='categorical',

                                                  subset='validation')



test_generator = test_datagen.flow_from_directory(test_data_dir,

                                                  target_size=(img_height, img_width),

                                                  batch_size=batch_size,

                                                  class_mode='categorical')
model = Sequential()

model.add(BatchNormalization(input_shape=(100, 100, 3)))

model.add(Conv2D(10, (4, 4), padding='same', activation='relu'))

model.add(Conv2D(10, (4, 4), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(BatchNormalization())

model.add(Conv2D(5, (4, 4), padding='same', activation='relu'))

model.add(Conv2D(5, (4, 4), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Conv2D(5, (4, 4), padding='same', activation='relu'))

model.add(Conv2D(5, (4, 4), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(118, activation='softmax'))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



history = model.fit_generator(train_generator,

                              steps_per_epoch=2000,

                              epochs=10,

                              validation_data=val_generator,

                              validation_steps=100)
x = model.evaluate_generator(generator=test_generator,

                             steps=100)

x
plt.plot(history.history['acc'], 

         label='Доля верных ответов на обучающем наборе')

plt.plot(history.history['val_acc'],

         label='Доля верных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля верных ответов')

plt.legend()

plt.show()