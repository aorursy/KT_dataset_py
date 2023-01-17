# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation,BatchNormalization

from keras import losses

from keras.optimizers import Adam, Adagrad

from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.model_selection import GridSearchCV

import keras

from keras.layers import LeakyReLU

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def VGG16():    

    model = Sequential()

    model.add(Conv2D(32,kernel_size = (3,3),strides=1,input_shape=(224,224,3),activity_regularizer = regularizers.l2(1e-8)))

    model.add(Activation("relu"))

    model.add(Conv2D(32,kernel_size = (3,3),strides=1,activity_regularizer = regularizers.l2(1e-8)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3),strides=2, padding='same', data_format=None))

    model.add(Dropout(0.25))



    model.add(Conv2D(64,kernel_size = (3,3),strides=1))

    model.add(Activation("relu"))

    model.add(Conv2D(64,kernel_size = (3,3),strides=1))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3),strides=2, padding='same', data_format=None))





    model.add(Conv2D(128,kernel_size = (3,3),strides=1))

    model.add(Activation("relu"))

    model.add(Conv2D(128,kernel_size = (3,3),strides=1,activity_regularizer = regularizers.l2(1e-8)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2),strides=2, padding='same', data_format=None))







    model.add(Flatten())



    model.add(Dense(4096,activity_regularizer = regularizers.l2(1e-8)))

    model.add(Activation("relu"))

    model.add(Dropout(0.1))







    model.add(Dense(512,activity_regularizer = regularizers.l2(1e-8)))

    model.add(Activation("relu"))

    model.add(Dropout(0.1))









    model.add(Dense(5,activation = 'softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

    model.summary()

    return model
from keras.preprocessing.image import ImageDataGenerator

EPOCHS=30

vggModel=VGG16()



train_datagen = ImageDataGenerator(

    rescale=1./255

)

test_datagen = ImageDataGenerator(rescale=1./255)





train_generator = train_datagen.flow_from_directory(

    '../input/train_resizedvgg net/',

    target_size=(224,224),

    batch_size=16

)

validation_generator = test_datagen.flow_from_directory(

        '../input/test_resizedvgg net/',

        target_size=(224,224),

        batch_size=32)

history=vggModel.fit_generator(

        train_generator,

        steps_per_epoch=2000,

        epochs=EPOCHS,

        validation_data=validation_generator,

        validation_steps=50

        )
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])





plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# set the matplotlib backend so figures can be saved in the background

# plot the training loss and accuracy

import sys

import matplotlib

print("Generating plots...")

sys.stdout.flush()

matplotlib.use("Agg")

matplotlib.pyplot.style.use("ggplot")

matplotlib.pyplot.figure()

N = EPOCHS

matplotlib.pyplot.plot(np.arange(0, N), history.history["loss"], label="train_loss")

matplotlib.pyplot.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

matplotlib.pyplot.plot(np.arange(0, N), history.history["acc"], label="train_acc")

matplotlib.pyplot.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")

matplotlib.pyplot.title("Training Loss and Accuracy on diabetic retinopathy detection")

matplotlib.pyplot.xlabel("Epoch #")

matplotlib.pyplot.ylabel("Loss/Accuracy")

matplotlib.pyplot.legend(loc="lower left")

matplotlib.pyplot.savefig("plot.png")
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
# serialize model to YAML

model_yaml = model.to_yaml()

with open("model.yaml", "w") as yaml_file:

    yaml_file.write(model_yaml)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")

 