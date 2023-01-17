# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
train_data_path="../input/chest-xray-pneumonia/chest_xray/train"

test_data_path="../input/chest-xray-pneumonia/chest_xray/test"

val_data_path="../input/chest-xray-pneumonia/chest_xray/val"

from keras.applications import VGG16

cov_base=VGG16(weights="imagenet",include_top=False,input_shape=(150,150,3))
cov_base.summary()
cov_base.trainable=False
cov_base.summary()
from keras import models

from keras import layers
model=models.Sequential()

model.add(cov_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,activation="relu"))

model.add(layers.Dense(1,activation="sigmoid"))
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,

                                 rotation_range=40,

                                 width_shift_range=0.2,

                                 shear_range=0.2,

                                 zoom_range=0.2,

                                 horizontal_flip=True,

                                 fill_mode="nearest"

                                 )
test_datagen=ImageDataGenerator(rescale=1./255)
train_data=train_datagen.flow_from_directory(train_data_path,target_size=(150,150),batch_size=100,class_mode="binary")
test_data=test_datagen.flow_from_directory(test_data_path,target_size=(150,150),batch_size=20,class_mode="binary")
model.compile(loss="binary_crossentropy",

              optimizer="adam",

              metrics=["acc"])
model.summary()
history=model.fit_generator(train_data,

                            steps_per_epoch=100,

                            epochs=6,

                            validation_data=test_data,

                            validation_steps=50)

training_loss=history.history["loss"]

training_aacuarcy=history.history["acc"]

val_acc=history.history["val_acc"]

val_loss=history.history["val_loss"]

plt.plot(np.arange(len(training_loss)),training_loss,color="red")

plt.plot(np.arange(len(val_loss)),val_loss,color="blue")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Epochs VS. loss")

plt.legend(["Training Loss","Validation Loss"])
plt.plot(np.arange(len(training_aacuarcy)),training_aacuarcy,color="red")

plt.plot(np.arange(len(val_acc)),val_acc,color="blue")

plt.xlabel("Epochs")

plt.ylabel("Accuarcy")

plt.title("Epochs VS. Accuarcy")

plt.legend(["Training Acc","Validation Acc"])