# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



img=mpimg.imread('../input/images/Images/n02085620-Chihuahua/n02085620_10074.jpg')

imgplot = plt.imshow(img)

plt.show()



classesList = ["n02085620-Chihuahua","n02085936-Maltese_dog","n02089867-Walker_hound","n02088466-bloodhound","n02091244-Ibizan_hound",

              "n02091032-Italian_greyhound","n02097474-Tibetan_terrier","n02100735-English_setter","n02102040-English_springer",

              "n02105505-komondor"]



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



image_generator = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        validation_split=0.2)



train = image_generator.flow_from_directory(

        '../input/images/Images/',

        target_size=(32, 32),

        batch_size=32,

        class_mode='categorical',

        classes = classesList,

        subset='training')



test = image_generator.flow_from_directory(

        '../input/images/Images/',

        target_size=(32, 32),

        batch_size=32,

        classes = classesList,

        class_mode='categorical',

        subset='validation')
import keras

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from pathlib import Path

from keras import backend as K

K.tensorflow_backend._get_available_gpus()





# Create a model and add layers

model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3), activation="relu"))

model.add(Conv2D(32, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax"))



# Compile the model

model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



model.fit_generator(

    train,

    steps_per_epoch=2000,

    epochs=50,

    validation_data=test)



# Save neural network structure

model_structure = model.to_json()

f = Path("model_structure.json")

f.write_text(model_structure)



# Save neural network's trained weights

model.save_weights("model_weights.h5")