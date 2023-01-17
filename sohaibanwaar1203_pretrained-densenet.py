# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

import tensorflow as tf

from keras.models import Sequential

import keras

print(os.listdir("../input"))

from keras.optimizers import Adam

# Any results you write to the current directory are saved as output.
dense_model=keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
dense_model.trainable = False

dense_model.summary()
model=Sequential()

 



# Add the vgg convolutional base model



model.add(dense_model)

 

# Add new layers

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))



model.add(keras.layers.Dense(4, activation='softmax'))

model.summary()


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

import PIL

epochs=30

train_datagen = ImageDataGenerator()



test_datagen = ImageDataGenerator()





train_generator = train_datagen.flow_from_directory( 

    '../input/final-dataset-dr/final model dataset/Categorical/train/',

    target_size=(224,224),

    batch_size=32

)

validation_generator = test_datagen.flow_from_directory( 

        '../input/final-dataset-dr/final model dataset/Categorical/test/',

        target_size=(224,224),

        batch_size=32)



modelhist = model.fit_generator(

        train_generator,

        steps_per_epoch=len(train_generator),

        epochs=epochs,

        validation_data=validation_generator,

        validation_steps=50

        )
# set the matplotlib backend so figures can be saved in the background

# plot the training loss and accuracy

import sys

import matplotlib

print("Generating plots...")

sys.stdout.flush()

matplotlib.use("Agg")

matplotlib.pyplot.style.use("ggplot")

matplotlib.pyplot.figure()

N = epochs 

matplotlib.pyplot.plot(np.arange(0, N),modelhist.history["loss"], label="train_loss")

matplotlib.pyplot.plot(np.arange(0, N), modelhist.history["val_loss"], label="val_loss")

matplotlib.pyplot.plot(np.arange(0, N), modelhist.history["acc"], label="train_acc")

matplotlib.pyplot.plot(np.arange(0, N),modelhist.history["val_acc"], label="val_acc")

matplotlib.pyplot.title("Cactus Image Classification")

matplotlib.pyplot.xlabel("Epoch #")

matplotlib.pyplot.ylabel("Loss/Accuracy")

matplotlib.pyplot.legend(loc="lower left")

matplotlib.pyplot.savefig("plot.png")
model_name = 'model.h5'

save_dir = os.getcwd()

# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)



# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)