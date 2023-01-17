# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.preprocessing.image  import ImageDataGenerator, img_to_array,load_img

import matplotlib.pyplot as plt

from glob import glob
train_path = "/kaggle/input/fruits/fruits-360/Training/"

test_path = "/kaggle/input/fruits/fruits-360/Test/"
img = load_img(train_path + "Apple Golden 1/0_100.jpg")

plt.imshow(img)

plt.title("Apple Golden")

plt.axis("off")

plt.show()
shape_of_image = img_to_array(img)

print(shape_of_image.shape)
classes = glob(train_path + "/*")

number_of_class = len(classes)

print("Number of class : " , number_of_class)
train_datagen = ImageDataGenerator(rescale = 1./255,

                   shear_range = 0.3,

                   horizontal_flip = True,

                   zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(train_path,

                                                   target_size = shape_of_image.shape[:2],

                                                   batch_size = 32,

                                                   color_mode = 'rgb',

                                                   class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(test_path,

                                                   target_size = shape_of_image.shape[:2],

                                                   batch_size = 32,

                                                   color_mode = 'rgb',

                                                   class_mode = 'categorical')
model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))

model.add(MaxPooling2D())



model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))

model.add(MaxPooling2D())



model.add(Conv2D(64,(3,3),activation = 'relu', input_shape = shape_of_image.shape))

model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(number_of_class,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',

              optimizer = 'rmsprop',

              metrics = ['accuracy'])
batch_size = 32

number_of_batch = 1600 // batch_size
hist = model.fit_generator(

    generator = train_generator,

    steps_per_epoch = number_of_batch,

    epochs = 100,

    validation_data = test_generator,

    validation_steps = 800 // batch_size

                   )
model.save_weights("trial.h5")
print(hist.history.keys())

plt.plot(hist.history["loss"],label = "Train Loss")

plt.plot(hist.history["val_loss"],label = "Validaton Loss")

plt.legend()

plt.show()
plt.figure()

plt.plot(hist.history["accuracy"],label = "Train Accuracy")

plt.plot(hist.history["val_accuracy"],label = "Validaton Accuracy")

plt.legend()

plt.show()
import json

with open("traial.json","w") as f:

    json.dump(hist.history,f)
import codecs 

with codecs.open("traial.json","r",encoding = "utf-8") as f:

    h = json.loads(f.read())