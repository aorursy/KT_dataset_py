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
#import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from shutil import copyfile
import os
import numpy as np 
import pandas as pd 
tf.__version__
# CONFIGURE GPUs
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.list_physical_devices('GPU'); print(gpus)
if len(gpus)==1: strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else: strategy = tf.distribute.MirroredStrategy()
import IPython.display as display
from PIL import Image
import pathlib
data_dir = pathlib.Path('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/')
image_count = len(list(data_dir.glob('*.jpg')))
image_count
df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
df.head(3)
df_sorted = df.sort_values(by='target', ascending=False)
df_sorted.head()
import shutil
ex = list(data_dir.glob('*.jpg'))

#for image_path in ex[:3]:
    #display.display(Image.open(str(image_path)))
str(ex[1])
# create a list of labels (0 - benign, 1 - cancer)
labels = []
filenames = []
counter = 0
for item in ex:
    tmp = df.loc[df['image_name'] == item.stem,'target'].iloc[0]
    labels.append(tmp)
    filenames.append(str(item))
    if tmp == 1:
        counter+=1
counter        
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_data = tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
next(iter(train_data))
# Function to load and preprocess each image
def _parse_fn(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label
IMAGE_SIZE = 224 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32
train_data = train_data.map(_parse_fn)
for image, label in train_data.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    #ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
train_ds_batched = prepare_for_training(train_data)
for image, label in train_data.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
#next(iter(train_ds_batched))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
METRICS = [
      #tf.keras.metrics.TruePositives(name='tp'),
      #tf.keras.metrics.FalsePositives(name='fp'),
      #tf.keras.metrics.TrueNegatives(name='tn'),
      #tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      #tf.keras.metrics.Precision(name='precision'),
      #tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.summary()
class_weight = {0: 1.,
                1: 1.}
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Accuracy()])
model.fit(train_ds_batched,
          epochs=20,
          class_weight=class_weight,
          steps_per_epoch = 100)
alt_model = Sequential()
alt_model.add(Conv2D(32,3, activation='relu', input_shape=(224, 224 ,3)))
alt_model.add(Dropout(0.5))
alt_model.add(MaxPooling2D())
alt_model.add(BatchNormalization())
alt_model.add(Conv2D(64,3, activation='relu'))
alt_model.add(Dropout(0.5))
alt_model.add(MaxPooling2D())
alt_model.add(BatchNormalization())
alt_model.add(Conv2D(128,3,activation='relu'))
alt_model.add(MaxPooling2D())
alt_model.add(Flatten())
alt_model.add(Dropout(0.5))
alt_model.add(BatchNormalization())
alt_model.add(Dense(512, activation='relu'))
alt_model.add(Dense(1,activation='softmax'))

alt_model.summary()
# Compile the model
alt_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
alt_model.fit(train_ds_batched,
          epochs=10,
          class_weight=class_weight,
          steps_per_epoch = 20)
#Using pretrained ImageNet
IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
# Pre-trained model with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
# Freeze the pre-trained model weights
base_model.trainable = True
# Trainable classification head
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
# Layer classification head with feature detector
model_mobileNet = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])


model_mobileNet.summary()
learning_rate = 0.0005
model_mobileNet.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
              loss='binary_crossentropy',
              metrics=METRICS
)
model_mobileNet.fit(train_ds_batched,
          epochs=5,
          steps_per_epoch = 50)
with strategy.scope():
    #Using pretrained ImageNet
    IMAGE_SIZE = 224
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # Pre-trained model with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pre-trained model weights
    base_model.trainable = True
    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    # Layer classification head with feature detector
    model_mobileNet = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    
    # Compile the model
    model_mobileNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_mobileNet.summary()
model_mobileNet.fit(train_ds_batched,
          epochs=5,
          steps_per_epoch = 50)