# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#       print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image  as mpimg

train=pd.read_csv("/kaggle/input/flower-recognition-he/he_challenge_data/data/train.csv")

test=pd.read_csv("/kaggle/input/flower-recognition-he/he_challenge_data/data/test.csv")

print(train.head(5))

print(train.groupby(['category']).count())

print("The training data frame: ",train.shape)

# Let us check some of the images

image = mpimg.imread("/kaggle/input/flower-recognition-he/he_challenge_data/data/train/1.jpg")

print(image.shape)

plt.imshow(image)

plt.show()

print("The flower is: ",train['category'][1])

image = mpimg.imread("/kaggle/input/flower-recognition-he/he_challenge_data/data/train/4.jpg")

plt.imshow(image)

plt.show()

print("The flower is: ",train['category'][4])
from keras.applications import VGG16

#Load the VGG model

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

from keras import layers

from keras import Model

#local_weights_file = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

image_size=500

vgg_conv = VGG16(weights='/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(image_size, image_size, 3))

#pre_trained_model=ResNet50(include_top = False, pooling = 'avg', weights = local_weights_file,input_shape=(500,500,3))

#for layer in pre_trained_model.layers:

#  layer.trainable = False

for layer in vgg_conv.layers[:-4]:

    layer.trainable = False

model = Sequential()

model.add(vgg_conv)

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(102, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 

              loss = 'categorical_crossentropy', 

              metrics = ['acc'])
#model = tf.keras.models.Sequential([

#tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 3)),

#    tf.keras.layers.MaxPooling2D(2, 2),

#    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

#    tf.keras.layers.MaxPooling2D(2, 2),

#    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

#    tf.keras.layers.MaxPooling2D(2, 2),

#    tf.keras.layers.Flatten(),

#    tf.keras.layers.Dropout(0.5),

#    tf.keras.layers.Dense(512, activation='relu'),

#    tf.keras.layers.Dense(102, activation='softmax')

#])



#model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers

from keras.applications.resnet50 import preprocess_input

import pandas as pd

import numpy as np

train['image_id']=train['image_id'].astype(str)

train['image_id']=train['image_id']+".jpg"

train['category']=train['category'].astype(str)

print(train.head(10))

test['image_id']=test['image_id'].astype(str)

test['image_id']=test['image_id']+".jpg"

print(test.head(10))

datagen=ImageDataGenerator(rescale=1./255.,preprocessing_function=preprocess_input,validation_split=0.10)
train_generator=datagen.flow_from_dataframe(

dataframe=train,

directory="/kaggle/input/flower-recognition-he/he_challenge_data/data/train/",

x_col="image_id",

y_col="category",

subset="training",

batch_size=32,

rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest',

seed=42,

shuffle=True,

class_mode="categorical",

target_size=(500,500))

valid_generator=datagen.flow_from_dataframe(

dataframe=train,

directory="/kaggle/input/flower-recognition-he/he_challenge_data/data/train/",

x_col="image_id",

y_col="category",

subset="validation",

batch_size=32,

seed=42,

shuffle=True,

class_mode="categorical",

target_size=(500,500))
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(

dataframe=test,

directory="/kaggle/input/flower-recognition-he/he_challenge_data/data/test/",

x_col="image_id",

y_col="category",

batch_size=1,

seed=42,

shuffle=False,

class_mode=None,

target_size=(500,500))
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=5)
submission=pd.read_csv("/kaggle/input/flower-recognition-he/he_challenge_data/data/sample_submission.csv")

print(submission.head(5))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()

pred=model.predict_generator(test_generator,

steps=STEP_SIZE_TEST,

verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

filenames=[f.split('.')[0] for f in filenames]

results=pd.DataFrame({"image_id":filenames,"category":predictions})

results = results.sort_values(by = ['image_id'], ascending = [True])

print(results.head(10))

results.to_csv("results.csv",index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='results.csv')