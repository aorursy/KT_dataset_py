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
import zipfile

zip_ref = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r')
zip_ref.extractall('/tmp/train')
zip_ref.close()
filenames = os.listdir("/tmp/train/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'label': categories
})
df.head()
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
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them

path="/tmp/train/train/"


# test=test.drop('label',1)
temp = []
for img_name in df.filename:
    image_path = path+img_name
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, 128 * 128)
    img = img.astype('float32')
    temp.append(img)


train_x = np.stack(temp)

train_x /= 255.0

train_x = train_x.reshape(len(df), 128,128,1).astype('float32')
train_y = to_categorical(df.label.values,num_classes=2)

print("no_prob")
train_x=np.array(train_x)
train_y=np.array(train_y)

x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2,random_state=4)
# time.sleep(10)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


aug.fit(train_x)
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
        model.add(Activation("sigmoid"))
 
        # return the constructed network architecture
        return model

EPOCHS = 50
INIT_LR = 1e-3
BS = 32
print("[INFO] compiling model...")
model = CNNmodel.build(width=128, height=128, depth=1, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),validation_data=(x_test, y_test),steps_per_epoch=len(x_train) // BS,epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
# model.save('model.h5')
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
zip_ref = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r')
zip_ref.extractall('/tmp/test1')
zip_ref.close()
path="/tmp/train/test1/"
c = 1
submission = pd.DataFrame(columns = ['id', 'label'])
for img in os.listdir(path):
    image = cv2.imread(path + img,0)
    # orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    predictions=model.predict(image)
    val = np.argmax(np.squeeze(predictions))
    submission = submission.append({'id': c,'label': val},ignore_index=True)

submission.to_csv('submission.csv' ,index=False)
print(submission.head())