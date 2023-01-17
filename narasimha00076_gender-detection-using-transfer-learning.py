# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import shutil

import glob

import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
import pandas as pd

data = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')

data.set_index('image_id',inplace = True)

data.head()
gender = data.Male
raw_train = []

raw_train_labels = []

raw_test = []

raw_test_labels = []
def get_images(data,no_of_images,labels,inp,outp):

    path = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'

    files = os.listdir(path)

    mcount, fcount = 0, 0

    for i in files[inp:outp]:

        if gender[i] == 1:

            mcount = mcount + 1

            if mcount == no_of_images:

                continue

        elif gender[i] == -1:

            gender[i] = 0

            fcount = fcount + 1

            if fcount == no_of_images:

                continue

        img = cv2.imread(os.path.join(path,i))

        data.append(img)

        labels.append(gender[i])

        if len(data) == 2 * no_of_images:

            return data, labels

        if len(data) % 100 == 0:

            print(len(data),'images')
raw_train,raw_train_labels = get_images(raw_train,2500,raw_train_labels,0,50000)

raw_test,raw_test_labels = get_images(raw_test,500,raw_test_labels,50000,100000)
len(raw_train),len(raw_train_labels),len(raw_test),len(raw_test_labels)
train_data = tf.data.Dataset.from_tensor_slices((raw_train,raw_train_labels))

test_data = tf.data.Dataset.from_tensor_slices((raw_test,raw_test_labels))
IMG_SIZE = 160

def format_example(image,label):

    image = tf.cast(image,dtype = tf.float32)

    image = (image / 255) - 1

    image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))

    return image, label

train_data = train_data.map(format_example)

test_data = test_data.map(format_example)
BATCH_SIZE = 32

SHUFFLE_BUFFER_SIZE = 2000

train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

test_data = test_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,include_top = False,weights = 'imagenet')

base_model.trainable = False
for image_batch, label_batch in train_data.take(1):

    pass

print(image_batch.shape)
print(base_model(image_batch).shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

print(global_average_layer(base_model(image_batch)).shape)
hidden_layer1 = tf.keras.layers.Dense(256,activation = 'relu')

dropout1 = tf.keras.layers.Dropout(0.5)

print(dropout1(hidden_layer1(global_average_layer(base_model(image_batch)))).shape)
hidden_layer2 = tf.keras.layers.Dense(128,activation = 'relu')

print(hidden_layer2(dropout1(hidden_layer1(global_average_layer(base_model(image_batch))))).shape)
prediction_layer = tf.keras.layers.Dense(1)

print(prediction_layer(hidden_layer2(dropout1(hidden_layer1(global_average_layer(base_model(image_batch)))))).shape)
model = tf.keras.Sequential([

     base_model,

     global_average_layer,

     hidden_layer1,

     dropout1,

     hidden_layer2,

     prediction_layer

])
base_learning_rate = 0.0001

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate),

              loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
num_train = 10000

num_test = 2500

initial_epochs = 20

steps_per_epochs = round(num_train) //  BATCH_SIZE

validation_steps = 4



loss0, accuracy0 = model.evaluate(test_data, steps = validation_steps)
callbacks = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/best_model.h5',save_best_only = True,monitor = 'val_accuracy',mode = max,verbose = 1)
history = model.fit(train_data,epochs = initial_epochs, validation_data = test_data,callbacks = [callbacks])
model.load_weights('best_model.h5')
model.evaluate(test_data)