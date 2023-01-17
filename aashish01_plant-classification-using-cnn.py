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
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam
def convert_to_array(image):
    try:
        im = cv2.imread(image)
        if im is None:
            return np.array([])
        else:
            im = cv2.resize(im, input_shape)
            return im
    except Exception as e:
        print(f"Error1 : {e}")
        return None
rootdir = 'plantvillage/PlantVillage/'

files = [i for i in os.listdir(rootdir)]
# print(files) #15 labels 

plant_image = []
plant_label = []

try:
    for file in files:
        im_dir = os.listdir(rootdir + file + '/')
        print(f"Processing...{file}")
        for image in im_dir[:400]:
            if image.endswith(".jpg") or image.endswith(".JPG"):
                image_dir = rootdir + file + '/'+image
                plant_image.append(np.array(convert_to_array(image_dir)))
                plant_label.append(file)
    print("Completed")
    
except Exception as e:
    print(f"Error2 : {e}")
plant_im = (np.array(plant_image, dtype='float32')-127.5)/127.5
l_binarizer = LabelBinarizer()
ima_label = l_binarizer.fit_transform(plant_label)
n_classes = len(l_binarizer.classes_)
x_train, x_test, y_train, y_test = train_test_split(plant_im, ima_label)
if K.image_data_format == "channels_first":
    input_shape = 3,256,256
else:
    input_shape = 256,256,3
model = Sequential()

model.add(Conv2D(16, (3, 3), activation = 'relu' ,padding="same",input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3),activation = 'relu' , padding="same"))
model.add(Conv2D(32, (3, 3),activation = 'relu' , padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),activation = 'relu' , padding="same"))
model.add(Conv2D(64, (3, 3),activation = 'relu' , padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3),activation = 'relu' , padding="same"))
model.add(Conv2D(128, (3, 3),activation = 'relu' , padding="same"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation = 'softmax'))

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size = 32,
                    epochs=25, 
                    verbose=1)