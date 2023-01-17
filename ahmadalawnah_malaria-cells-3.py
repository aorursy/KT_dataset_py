# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization

from keras.callbacks.callbacks import History

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
WIDTH = 128

HEIGHT = 128

CHANNELS = 3

batch_size = 64;
path_training= '../input/cell-images-for-detecting-malaria/Training/'

path_testing= '../input/cell-images-for-detecting-malaria/Testing/'



training_generator = ImageDataGenerator(validation_split=0.2,rescale=1./255)

print("Training: ")

images_training = training_generator.flow_from_directory(path_training, target_size=(WIDTH, HEIGHT), classes=['Parasitized','Uninfected'], subset='training', batch_size=batch_size)

print("Validation: ")

images_validation = training_generator.flow_from_directory(path_training, target_size=(WIDTH, HEIGHT), classes=['Parasitized','Uninfected'], subset = 'validation', batch_size=batch_size)



testing_generator = ImageDataGenerator(rescale=1./255)

print("Testing: ")

images_testing = testing_generator.flow_from_directory(path_testing,target_size=(WIDTH, HEIGHT), class_mode = None, batch_size=batch_size, shuffle=True)

images_testing_with_labels = testing_generator.flow_from_directory(path_testing,target_size=(WIDTH, HEIGHT), classes=['Parasitized','Uninfected'], batch_size=batch_size, shuffle=True)
model = Sequential()



model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, CHANNELS))) 

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))



model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = History()

epochs = 10

result = model.fit_generator(images_training, epochs=epochs, validation_data = images_validation, steps_per_epoch=len(images_training), validation_steps = len(images_validation), callbacks=[history])
plt.plot(history.history['accuracy'], color='blue', label="Training accuracy")

plt.plot(history.history['val_accuracy'],color='red', label="Validation accuracy")

plt.xticks(np.arange(1,epochs,1))

plt.legend(loc='best',shadow=True)

plt.show()

result = model.evaluate_generator(images_testing_with_labels)
acc = result[1]*100

acc = np.round(acc, decimals = 2)

print("Accuracy: "+ str(acc)+'%')
x,y = images_testing_with_labels.next()

labels = y;

sample = x;
samples = model.predict_classes(sample)
actual_labels = list()

for i in range(len(labels)):

    if labels[i][0] == 0.0:

        actual_labels.append(1)

    else:

        actual_labels.append(0)


for i in range(32):

    img  = sample[i]

    pred = "uninfected"

    actual = "uninfected"

    if samples[i]==0:

        pred = "infected"

    if (actual_labels[i]==0):

        actual = "infected"

    plt.imshow(img)

    plt.title("Predicted: "+ pred+", Actual: "+ actual)

    plt.show()