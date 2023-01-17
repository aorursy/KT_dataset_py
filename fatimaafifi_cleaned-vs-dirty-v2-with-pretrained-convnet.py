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
!pip install tf-nightly
import os



import numpy as np



import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
! unzip /kaggle/input/platesv2/plates.zip

! ls 

import tensorflow_datasets as tfds

tfds.disable_progress_bar()
image_size = (200, 200)

batch_size = 10
raw_train = tf.keras.preprocessing.image_dataset_from_directory(

    "plates/train",

    validation_split=0.3,

    subset="training",

    seed=1307,

    image_size=image_size,

    batch_size=batch_size,

)

raw_validation = tf.keras.preprocessing.image_dataset_from_directory(

    "plates/train",

    validation_split=0.3,

    subset="validation",

    seed=1307,

    image_size=image_size,

    batch_size=batch_size,

)

raw_train
raw_validation
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 10))

for images, labels in raw_train.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(int(labels[i]))

        plt.axis("off")
IMG_SIZE = 200 # All images will be resized to 160x160



def format_example(image, label):

    image = tf.cast(image, tf.float32)

    image = (image/127.5) - 1

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label
train = raw_train.map(format_example)

validation = raw_validation.map(format_example)
train
validation
BATCH_SIZE = 32

SHUFFLE_BUFFER_SIZE = 20
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

validation_batches = validation.batch(BATCH_SIZE)

train_batches
validation_batches
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=15,

                    width_shift_range=.1,

                    height_shift_range=.1,

                    horizontal_flip=True,

                    zoom_range=0.1, 

                    brightness_range=[0.8,1.0]

                    )
PATH =  'plates'

train_ds = os.path.join(PATH, 'train')

train_ds
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                     directory=train_ds,

                                                     shuffle=True,

                                                     target_size=(IMG_SIZE, IMG_SIZE),

                                                     class_mode='binary', 

)
sample_train, _ = next(train_data_gen)

plotImages(sample_train[:5])
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)



# Create the base model from the pre-trained model MobileNet V2

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')

base_model.trainable = False
# Let's take a look at the base model architecture

base_model.summary()
image_batch = sample_train[:batch_size]

image_batch.shape
feature_batch = base_model(image_batch)

print(feature_batch.shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)

print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(1)

prediction_batch = prediction_layer(feature_batch_average)

print(prediction_batch.shape)
model = tf.keras.Sequential([

  base_model,

  global_average_layer,

  prediction_layer

])
base_learning_rate = 0.0001

model.compile(optimizer=tf.optimizers.RMSprop(lr=base_learning_rate),

              loss=tf.losses.BinaryCrossentropy(from_logits=True),

              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
model.summary()
len(model.trainable_variables)

initial_epochs = 10

validation_steps=20



test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(  

        'plates',

        classes=['test'],

        target_size = (IMG_SIZE, IMG_SIZE),

        batch_size = 1,

        shuffle = False,        

        class_mode = None)  
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
sub_df.to_csv('sub.csv', index=False)

print('done!!!')