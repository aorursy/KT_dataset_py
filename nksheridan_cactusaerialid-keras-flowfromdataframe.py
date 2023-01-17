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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers



import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
traindf=pd.read_csv('../input/aerial-cactus-identification/train.csv',dtype=str)

testdf=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv',dtype=str)
traindf.head(5)
traindf.has_cactus=traindf.has_cactus.astype(str)
print('out dataset has {} rows and {} columns'.format(traindf.shape[0],traindf.shape[1]))
traindf['has_cactus'].value_counts()
import zipfile

Dataset = "train"


with zipfile.ZipFile("../input/aerial-cactus-identification/"+Dataset+".zip","r") as z:
    z.extractall(".")
    

print(os.listdir("../working/"))
from IPython.display import Image
from keras.preprocessing import image
Image(os.path.join("../working/train/",traindf.iloc[0,0]),width=250,height=250)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../working/train/",
x_col="id",
y_col="has_cactus",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(150,150))
valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../working/train/",
x_col="id",
y_col="has_cactus",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(150,150))
Dataset = "test"


with zipfile.ZipFile("../input/aerial-cactus-identification/"+Dataset+".zip","r") as z:
    z.extractall(".")
    
print(os.listdir("../working/"))
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="../working/test/",
x_col="id",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(150,150))
from keras import layers, models
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, ZeroPadding2D
from keras.models import Model

import keras.backend as K
from keras.models import Sequential
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
model.summary()

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=valid_generator,validation_steps=50)