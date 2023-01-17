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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
df_test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
df_train.head()
df_test.head()
print("train",df_train.shape)
print("test", df_test.shape)
train_y = df_train['label']
df_train.drop('label', 1, inplace = True)
print("train", df_train.shape)
train_y.values
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input
import pandas as pd
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import cv2
import time
import h5py
train_x = np.array(df_train)

train_x = train_x.reshape(len(df_train), 28, 28,1).astype('float32')
train_y = to_categorical(train_y.values,num_classes=10)
class CNNmodel:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
 
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
 
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        # return the constructed network architecture
        return model
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
train_x=np.array(train_x)
train_y=np.array(train_y)
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2,random_state=4)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


aug.fit(train_x)
model = CNNmodel.build(width=28, height=28, depth=1, classes=10)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),validation_data=(x_test, y_test),steps_per_epoch=len(x_train) // BS,epochs=EPOCHS, verbose=1)

def plot_graph(H,EPOCHS,INIT_LR,BS):

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on our system")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # plt.savefig(args["plot"])
    plt.show()
plot_graph(H,EPOCHS,INIT_LR,BS)
df_test.head()
y_actu = df_test['label']
df_test.drop('label', 1, inplace = True)
df_test.shape

test_x = np.array(df_test)
test_x = test_x.reshape(len(test_x), 28, 28, 1)
y_pred = []
for i in range(len(test_x)):
    test_x[i] = test_x[i].astype("float32")
    val = np.expand_dims(test_x[i], axis=0)
    predictions = model.predict(val)
    y_pred.append(np.argmax(np.squeeze(predictions)))
print("type", type(y_pred), "len", len(y_pred))
y_pred = np.array(y_pred)
print(y_pred[2], y_actu[2])
c = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_actu[i]:
        c += 1
print(c / len(y_pred))
