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
!pip install imutils
# import necessary modules
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils import paths
import cv2
import os
# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # initialize the model along with the input shape to be
        # "channedls last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # BLOCK #1
        model.add(Conv2D(96, (11, 11), strides = (4, 4),
                        input_shape=inputShape, padding="same",
                        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))

        # BLOCK #2
        model.add(Conv2D(256, (5,5), padding="same",
                        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # BLOCK #3
        model.add(Conv2D(384, (3, 3), padding="same",
                        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding="same",
                        kernel_regulerizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3,3), padding="same",
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))

        # BLOCK #4
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=le(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # BLOCK #5
        model.add(Dense(4096, kernel_regularizer=le(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))
# load data and preprocess
data = []
labels = []

with tf.device('/GPU:0'):
    for path in imutils.paths.list_images('/kaggle/input'):
        image = cv2.imread(path)
        image = imutils.resize(image, width=300)
        array = img_to_array(image)
        data.append(array)

        label = path.split(os.path.sep)[-2]
        labels.append(label)

# scale data
data = data / 255.0
labels = np.array(label)

# split data to train-test
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.2, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize the model
print("Compiling model...")
opt = SGD(le=1e-3)
model = AlexNet.build(width, height, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
             metrics=["accuracy"])

# train the model
print("Training model...")
with tf.device('/GPU:0'):
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
         batch_size=16, epochs=100, verbose=1)

# evaluzate the model
print("Evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1)))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# train model
#evaluate model
