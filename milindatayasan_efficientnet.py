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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import os
import gc
print(os.listdir("../input"))
data_fer = pd.read_csv('../input/fer2013/fer2013.csv')
data_fer.head()
!pip install -U git+https://github.com/qubvel/efficientnet
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image


from efficientnet.keras import EfficientNetB0 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
data = pd.read_csv('../input/fer2013/fer2013.csv')
data.head()
print("Datanın satır ve sütün sayıları = ", data.shape)
print("Sütünların ismi = ", data.columns)

data_fer = pd.read_csv('../input/fer2013/fer2013.csv')
data_fer.head()
training = data.loc[data["Usage"] == "Training"]
public_test = data.loc[data["Usage"] == "PublicTest"]
private_test = data.loc[data["Usage"] == "PrivateTest"]

print("Traning Data = ", training.shape)
print("public test Data = ", public_test.shape)
print("Private test Data = ", private_test.shape)
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
idx_to_emotion_fer = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

X_fer_train, y_fer_train = np.rollaxis(data_fer[data_fer.Usage == "Training"][["pixels", "emotion"]].values, -1)
X_fer_train = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_train]).reshape((-1, 48, 48))
y_fer_train = y_fer_train.astype('int8')

X_fer_test_public, y_fer_test_public = np.rollaxis(data_fer[data_fer.Usage == "PublicTest"][["pixels", "emotion"]].values, -1)
X_fer_test_public = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_public]).reshape((-1, 48, 48))
y_fer_test_public = y_fer_test_public.astype('int8')

X_fer_test_private, y_fer_test_private = np.rollaxis(data_fer[data_fer.Usage == "PrivateTest"][["pixels", "emotion"]].values, -1)
X_fer_test_private = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_private]).reshape((-1, 48, 48))
y_fer_test_private = y_fer_test_private.astype('int8')

print(f"X_fer_train shape: {X_fer_train.shape}; y_fer_train shape: {y_fer_train.shape}")
print(f"X_fer_test_public shape: {X_fer_test_public.shape}; y_fer_test_public shape: {y_fer_test_public.shape}")
print(f"X_fer_test_private shape: {X_fer_test_private.shape}; y_fer_test_private shape: {y_fer_test_private.shape}")
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Concatenate,Dropout
from keras.utils import to_categorical
import tensorflow as tf
def one_hot(y):
    return to_categorical(y, 7)
width = 48
height = 48
dropout_rate = 0.2
#input_shape = (height, width, 1)
input_shape1 = Input(shape=(height,width,1))
input_shape = Concatenate()([input_shape1, input_shape1, input_shape1]) 
conv_base = Net(weights='imagenet', include_top=False,input_shape=(48, 48, 3))
conv_output = conv_base(input_shape)
conv_output_flattened = Flatten()(conv_output)
dense_out = Dense(48, activation='relu')(conv_output_flattened)
dense_out= Dropout(0.2)(dense_out)

out = Dense(7, activation='softmax')(dense_out)

model = Model(inputs=input_shape1, outputs=out)


model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])
hist=model.fit(
    X_fer_train.reshape((-1, 48, 48, 1)), 
    one_hot(y_fer_train), 
    batch_size=128, 
    epochs=8, 
    validation_data=(X_fer_test_public.reshape((-1,48, 48, 1)), one_hot(y_fer_test_public)))


acc = hist.history["categorical_accuracy"]
val_acc = hist.history["val_categorical_accuracy"]
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label = "Train Accuracy")
plt.plot(epochs, val_acc, "b", label = "Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, "bo", label = "Train Loss")
plt.plot(epochs, val_loss, "b", label = "Validation Loss")
plt.legend()


plt.show()

from keras.models import load_model

model.save('efficientetB0.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
model.save_weights('efficientetB0_weights.h5')
# returns a compiled model
# identical to the previous one
model = load_model('efficientetB0.h5')
model.load_weights('efficientetBl_weights.h5', by_name=True)
#conv_base.summary()
acc = hist.history["categorical_accuracy"]
val_acc = hist.history["val_categorical_accuracy"]
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label = "Train Accuracy")
plt.plot(epochs, val_acc, "b", label = "Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, "bo", label = "Train Loss")
plt.plot(epochs, val_loss, "b", label = "Validation Loss")
plt.legend()


plt.show()
from keras.models import load_model

model.save('efficientetB0.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
model.save_weights('efficientetB0_weights.h5')
# returns a compiled model
# identical to the previous one
model = load_model('efficientetB0.h5')
model.load_weights('efficientetBl_weights.h5', by_name=True)