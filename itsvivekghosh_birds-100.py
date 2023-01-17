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
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenetV2 import MobileNetV2
import tensorflow as tf
import tensorflow.keras
import os
BASE_DIR = '../input/100-bird-species/'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')
CATEGORIES = os.listdir(TRAIN_DIR)
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet

train_data = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
)

validation_data = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

test_data = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)
conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False
import keras
x=keras.layers.Flatten()(conv_base.output)
prediction=keras.layers.Dense(len(CATEGORIES), activation='softmax')(x)
model=keras.models.Model(inputs=conv_base.input, outputs=prediction)
model.compile(
    optimizer=tensorflow.keras.optimizers.RMSprop(lr=0.001),
    loss=tensorflow.keras.losses.categorical_crossentropy,
    metrics=['accuracy'],
)
model.summary()
history = model.fit_generator(
    train_data,
    steps_per_epoch=280,
    validation_data=validation_data,
    validation_steps=14,
    epochs=8,
)
from keras.preprocessing import image

img=image.load_img('/kaggle/input/test/ALEXANDRINE PARAKEET/1.jpg', target_size=(224, 224))

import numpy as np

x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
ans=classes.argmax()
print(ans)