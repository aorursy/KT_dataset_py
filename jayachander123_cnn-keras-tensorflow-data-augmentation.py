# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os,shutil

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
base_dir = "../input"
train_dir = "../input/train/train"

test_dir = "../input/test1/test1"
train_imagenames = os.listdir("../input/train/train")

image_name = []

image_paths = []

for imagename in train_imagenames:

    image = imagename.split('.')[0]

    image_path = os.path.join(train_dir, imagename)

    image_paths.append(image_path)

    image_name.append(image)

    

    





image_df = pd.DataFrame()



image_df["image_path"] = image_paths

image_df["image_name"] = image_name



image_df.head()
image_df["image_name"].value_counts()
from keras import layers

from keras import models

import tensorflow as tf
model = tf.keras.models.Sequential([

    

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.summary()
from keras import optimizers

from keras.optimizers import SGD

from tensorflow.keras.optimizers import RMSprop

from sklearn.preprocessing import LabelEncoder



model.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=1e-4), metrics = ["acc"])
from sklearn.model_selection import train_test_split

train, validation = train_test_split(image_df,test_size=0.2, random_state=42) 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 40, width_shift_range = 0.2,

                                   height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_dataframe(train, x_col = "image_path", y_col = "image_name", target_size = (150, 150), batch_size = 128, class_mode = "binary")

validation_generator = validation_datagen.flow_from_dataframe(validation, x_col = "image_path", y_col = "image_name", target_size = (150, 150), batch_size = 128, class_mode = "binary")

#class myCallback(tf.keras.callbacks.Callback):

  #def on_epoch_end(self, epoch, logs={}):

    #if(logs.get('acc')>0.9):

    #        print("\nReached 60% accuracy so cancelling training!")

    #self.model.stop_training = True

#callbacks = myCallback()
history = model.fit_generator(train_generator,

                             steps_per_epoch = 80, epochs = 40, validation_data = validation_generator, validation_steps = 200)
import matplotlib.pyplot as plt

acc = history.history["acc"]

val_acc = history.history["val_acc"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, "bo", label = "Training acc")

plt.plot(epochs, val_acc, "b", label = "Validation acc")

plt.title("Training and Validation accuracy")

plt.legend()

plt.figure()

plt.plot(epochs, loss, "bo", label = "Training loss")

plt.plot(epochs, val_loss, "b", label = "Validation loss")

plt.title("Training and Validation loss")

plt.legend()

plt.show()
test_datagen = ImageDataGenerator(rescale = 1./255)

test_dir = "../input/test1"



#test_generator = test_datagen.flow_from_directory(test_dir, target_size =(150, 150), batch_size = 80, class_mode = None, shuffle = False)



test_generator = test_datagen.flow_from_directory(

    directory= test_dir,

    target_size=(150, 150),

    color_mode="rgb",

    batch_size=100,

    class_mode=None,

    shuffle=False

)

results = model.predict_generator(test_generator, steps = 125)
results
Final_df =[1 if value>0.5 else 0 for value in results]



submission = pd.DataFrame({"label": Final_df})
submission