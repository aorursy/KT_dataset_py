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
        os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
train_path = "../input/fruits/fruits-360/Training/"
test_path = "../input/fruits/fruits-360/Test/"
img = load_img(train_path + "Avocado/104_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
x = img_to_array(img)
print(x.shape) # x, y, rgb
class_name = glob(train_path + "/*")
number_of_class = len(class_name)
print("Number of class: ", number_of_class)
# Cnn Model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_class)) #output
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 32 # image number in every iteration
# Data Generation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.3,# shear_range = random dönüyor
                                   horizontal_flip=True, # horizontol_flip = horizontal olarak yan döndü 
                                   zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=x.shape[:2],
                                                    batch_size=batch_size,
                                                    color_mode="rgb",
                                                    class_mode="categorical")
test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=x.shape[:2],
                                                    batch_size=batch_size,
                                                    color_mode="rgb",
                                                    class_mode="categorical")

# Train - Test

hist = model.fit_generator(generator=train_generator, 
                    steps_per_epoch=1600//batch_size, 
                    epochs=100, 
                    validation_data=test_generator,
                    validation_steps=800//batch_size)

# model save
model.save_weights("fruit.h5")
# model evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"], label="Train accuracy")
plt.plot(hist.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()
# save history
import json
with open("cnn_fruit_hist.json", "w") as f:
    json.dump(str(hist.history), f)
# Load history
import codecs
with codecs.open("cnn_fruit_hist.json", "r", encoding='utf-8') as f:
    h = json.loads(f.read())
