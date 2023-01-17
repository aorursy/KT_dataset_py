!pip install tensorflow==2.3.0 -q

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
import PIL


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:',tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_systems(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

print(tf.__version__)
#!/usr/bin/python

import os, sys

# Create new Train and val folders

base_dir = 'kaggle/input/RiceLeafs'
train_path = '/kaggle/input/RiceLeafs/train'
val_path = 'kaggle/input/RiceLeafs/validation/'

column_names = os.listdir(train_path)
for i in column_names:
    os.makedirs(f'../kaggle/output/train/{i}')
    os.makedirs(f'../kaggle/output/validation/{i}')

out_path = '../kaggle/output/train/'



from PIL import Image
def resize(input_path,folder,column_name):
    dirs = os.listdir(input_path)
    for item in dirs:
        item_path = input_path +'/' +item
        if os.path.isfile(item_path):
            #print('CHECK')
            im = Image.open(item_path)

            # Check whether the specified 
            # path exists or not 
            outpath = f'/kaggle/kaggle/output/{folder}/{column_name}'
            temp_out_path = outpath+'/'+item
            f, e = os.path.splitext(temp_out_path)

            imResize = im.resize((150,150), Image.ANTIALIAS)
            #print('CHECK 3')
            imResize.save(f + '.jpg', 'JPEG', quality=90)




input_path = '../input/RiceLeafs/train/Healthy'
folder = 'train'
column_name = 'Healthy'
resize(input_path,folder,column_name)

input_path = '../input/RiceLeafs/train/BrownSpot'
folder = 'train'
column_name = 'BrownSpot'
resize(input_path,folder,column_name)

input_path = '../input/RiceLeafs/train/Hispa'
folder = 'train'
column_name = 'Hispa'
resize(input_path,folder,column_name)

input_path = '../input/RiceLeafs/train/LeafBlast'
folder = 'train'
column_name = 'LeafBlast'
resize(input_path,folder,column_name)

print('Done with train resizing')
## VALIDATION
input_path = '../input/RiceLeafs/validation/Healthy'
folder = 'validation'
column_name = 'Healthy'
resize(input_path,folder,column_name)

input_path = '../input/RiceLeafs/validation/BrownSpot'
folder = 'validation'
column_name = 'BrownSpot'
resize(input_path,folder,column_name)

input_path = '../input/RiceLeafs/validation/Hispa'
folder = 'validation'
column_name = 'Hispa'
resize(input_path,folder,column_name)

input_path = '../input/RiceLeafs/validation/LeafBlast'
folder = 'validation'
column_name = 'LeafBlast'
resize(input_path,folder,column_name)

print('Done with Validation resizing')
os.path.exists('/kaggle/kaggle/output/validation/Healthy/')
os.path.exists('/kaggle/kaggle/output/train/')
os.path.exists('/kaggle/kaggle/output/validation/')

os.listdir('/kaggle/kaggle/output/train/BrownSpot/')
data_dir = os.path.join(os.path.dirname('/kaggle/kaggle/'), 'output')
# Use this if you avoided the resizing
#data_dir = os.path.join(os.path.dirname('../input/'), 'riceleafs/RiceLeafs')
train_dir = os.path.join(data_dir, 'train')
train_BrownSpot_dir = os.path.join(train_dir, 'BrownSpot')
train_Healthy_dir = os.path.join(train_dir, 'Healthy')
train_Hispa_dir = os.path.join(train_dir, 'Hispa')
train_LeafBlast_dir = os.path.join(train_dir, 'LeafBlast')


validation_dir = os.path.join(data_dir, 'validation')
validation_BrownSpot_dir = os.path.join(validation_dir, 'BrownSpot')
validation_Healthy_dir = os.path.join(validation_dir, 'Healthy')
validation_Hispa_dir = os.path.join(validation_dir, 'Hispa')
validation_LeafBlast_dir = os.path.join(validation_dir, 'LeafBlast')
train_BrownSpot_names = os.listdir(train_BrownSpot_dir)
print(train_BrownSpot_names[:10])

train_Healthy_names =  os.listdir(train_Healthy_dir)
print(train_Healthy_names[:10])

train_Hispa_names = os.listdir(train_Hispa_dir)
print(train_Hispa_names[:10])

train_LeafBlast_names =  os.listdir(train_LeafBlast_dir)
print(train_LeafBlast_names[:10])

import time
import os
from os.path import exists

def count(dir, counter=0):
    "returns number of files in dir and subdirs"
    for pack in os.walk(dir):
        for f in pack[2]:
            counter += 1
    return dir + " : " + str(counter) + " files"

print('total images for training :', count(train_dir))
print('total images for validation :', count(validation_dir))
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Parameters for our graph; we'll outpu images in a 4x4 configuration
nrows = 4
ncols = 4

# for iternating over images
pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()
fig.set_size_inches(ncols *4, nrows*4)

pic_index += 4
next_BrownSpot_pix = [os.path.join(train_BrownSpot_dir, fname)
                for fname in train_BrownSpot_names[pic_index-4:pic_index]]

next_Healthy_pix = [os.path.join(train_Healthy_dir, fname)
                for fname in train_Healthy_names[pic_index-4:pic_index]]

next_Hispa_pix = [os.path.join(train_Hispa_dir, fname)
                for fname in train_Hispa_names[pic_index-4:pic_index]]

next_LeafBlast_pix = [os.path.join(train_LeafBlast_dir, fname)
                for fname in train_LeafBlast_names[pic_index-4:pic_index]]

for i, img_path in enumerate(next_BrownSpot_pix + next_Healthy_pix + next_Hispa_pix + next_LeafBlast_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows,ncols,i +1)
  #sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 100
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
print(strategy.num_replicas_in_sync)
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_ds = ImageDataGenerator(
# rescale = 1./255,
# width_shift_range= 0.2,
# height_shift_range = 0.2,
# shear_range = 0.4,
# rotation_range = 40,
# horizontal_flip = True,
# fill_mode = 'nearest')

training_generator = image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 220,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
    #class_mode = 'categorical'
)

# validation_ds = ImageDataGenerator(
#             rescale = 1./255)

validation_generator = image_dataset_from_directory(
        validation_dir,
        validation_split = 0.2,
        subset = 'validation',
        seed = 220,
        batch_size = BATCH_SIZE,
        image_size = IMAGE_SIZE,
        )
plt.figure(figsize=(10, 10))
for images, labels in training_generator.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(training_generator.class_names[labels[i]])
    plt.axis("off")
class_names = os.listdir(train_dir)

print(class_names)

training_generator.class_names = class_names
validation_generator.class_names = class_names

NUM_CLASSES = len(class_names)
def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_ds = training_generator.map(one_hot_label, num_parallel_calls=AUTOTUNE)
val_ds = validation_generator.map(one_hot_label, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters,3,activation = 'relu',padding = 'same'),
        tf.keras.layers.SeparableConv2D(filters,3,activation = 'relu',padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])
    return block
def dense_block(units,dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units,activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    return block
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE,3)),
        tf.keras.layers.Conv2D(16,3,activation = 'relu', padding = 'same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        #conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        
        #dense_block(512,0.7),
        dense_block(128,0.5),
        dense_block(64,0.3),
        
        tf.keras.layers.Dense(NUM_CLASSES,activation = 'softmax')
            ])
    return model

with strategy.scope():
    model = build_model()
    
    METRICS = [tf.keras.metrics.AUC(name='auc')]
    
    model.compile(
    optimizer = 'adam',
    loss = tf.losses.CategoricalCrossentropy(),
    metrics = METRICS
    )
def exponential_decay(lr0,s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch/s) 
    return exponential_decay_fn


exponential_decay_fn = exponential_decay(0.001,20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('Alzheimer_model.h5',
                                                  save_best_only = True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10,
                                                    restore_best_weights = True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=EPOCHS
)
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['auc', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])