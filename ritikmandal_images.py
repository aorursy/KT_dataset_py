###   Data Loaded properly  ###
import numpy as np
import pandas as pd
import pathlib
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import urllib
from tensorflow.keras import layers
import os
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.__version__
test_data_dir = pathlib.Path("/kaggle/input/kermany2018/OCT2017 /test/")
train_data_dir = pathlib.Path("/kaggle/input/kermany2018/OCT2017 /train/")
validate_data_dir = pathlib.Path("/kaggle/input/kermany2018/OCT2017 /val/")
test_image_count = len(list(test_data_dir.glob("*/*.jpeg")))
print("Test Image Count:", test_image_count)

train_image_count = len(list(train_data_dir.glob("*/*.jpeg")))
print("Train Image Count:",train_image_count)

validate_image_count = len(list(validate_data_dir.glob("*/*.jpeg")))
print("Validate Image Count:",validate_image_count)
CLASS_NAMES = np.array([item.name for item in validate_data_dir.glob('*')])
print(CLASS_NAMES)
test_data = list(test_data_dir.glob('*/*'))
train_data = list(train_data_dir.glob('*/*'))
validate_data = list(validate_data_dir.glob('*/*'))
#just to check the images

for image_path in test_data[:3]:
    display.display(Image.open(str(image_path)))
    
for image_path in train_data[:3]:
    display.display(Image.open(str(image_path)))
    
for image_path in validate_data[:3]:
    display.display(Image.open(str(image_path)))
list_test_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'))
list_train_ds = tf.data.Dataset.list_files(str(train_data_dir/'*/*'))
list_validate_ds = tf.data.Dataset.list_files(str(validate_data_dir/'*/*'))
#this too to check the images

for f in list_test_ds.take(5):
    print(f.numpy())
    
for f in list_train_ds.take(5):
    print(f.numpy())

for f in list_validate_ds.take(5):
    print(f.numpy())
boxes = [[0,0,0.6,0.6],[0,0.4,0.6,1],[0.4,0,1,0.6],[0.4,0.4,1,1]]  #This NEEDS TUNING ## normalized coordinates for the partition of the image. each box coordinates corresponds to each partition as [y1,x1,y2,x2]## change it as suitable ##
box_indices = [0]
crop_size = [128, 128]                                           ###THIS NEEDS TUNING AS WELL ## this is the final size of each of the partition of the image ## change it as suitable  ###
    
def partition(img):
    image = tf.expand_dims(img, 0)
    final_images = tf.image.crop_and_resize( image , box, box_indices, crop_size, method='bilinear', extrapolation_value=0,)
    return final_images

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
#    img = tf.image.resize(img, [128,128], method='nearest')    #############
    img = partition(img)
    return img, label
# For part 1

box = [boxes[0]]               ## upper left box

labeled_test_ds_1 = list_test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_train_ds_1 = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_validate_ds_1 = list_validate_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# For part 2
## don't forget to change the box just before training....## important

box = [boxes[1]]               ## upper right box

labeled_test_ds_2 = list_test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_train_ds_2 = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_validate_ds_2 = list_validate_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# For part 3
## don't forget to change the box just before training....## important

box = [boxes[2]]               ## bottom left box

labeled_test_ds_3 = list_test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_train_ds_3 = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_validate_ds_3 = list_validate_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# For part 4
## don't forget to change the box just before training....## important

box = [boxes[3]]               ## bottom right box

labeled_test_ds_4 = list_test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_train_ds_4 = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_validate_ds_4 = list_validate_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for image, label in labeled_test_ds_1.take(1):
    
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    print("Upper left")
    img = np.squeeze(image.numpy())
    plt.imshow(img)
    plt.show()
    
for image, label in labeled_test_ds_2.take(1):
    
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    print("Upper right")
    img = np.squeeze(image.numpy())
    plt.imshow(img)
    plt.show()
    
for image, label in labeled_test_ds_3.take(1):
    
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    print("bottom left")
    img = np.squeeze(image.numpy())
    plt.imshow(img)
    plt.show()
    
for image, label in labeled_test_ds_4.take(1):
    
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    print("bottom right")
    img = np.squeeze(image.numpy())
    plt.imshow(img)
    plt.show()
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()

    ds = ds.batch(100)                                     ######     Whats the batch size  ####

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
train_ds_1 = prepare_for_training(labeled_train_ds_1)
test_ds_1 = prepare_for_training(labeled_test_ds_1)

train_ds_2 = prepare_for_training(labeled_train_ds_2)
test_ds_2 = prepare_for_training(labeled_test_ds_2)

train_ds_3 = prepare_for_training(labeled_train_ds_3)
test_ds_3 = prepare_for_training(labeled_test_ds_3)

train_ds_4 = prepare_for_training(labeled_train_ds_4)
test_ds_4 = prepare_for_training(labeled_test_ds_4)

image_batch, label_batch = next(iter(train_ds_1))
print(train_ds_1)
print(type(train_ds_1))
show_batch(np.squeeze(image_batch.numpy()), label_batch.numpy())








from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    return image, label

def augment(image,label):
    image,label = convert(image, label)
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding

    return image,label
BATCH_SIZE = 64
# Only use a subset of the data so it's easier to overfit, for this tutorial
NUM_EXAMPLES = 2048
augmented_train_batches = (
    train_ds
    .take(NUM_EXAMPLES)             ###########################################
    .cache()
    .shuffle(train_image_count//4)
    # The augmentation is added here.
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
) 
validation_batches = (
    test_ds
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(2*BATCH_SIZE)
)
def make_model():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer = 'adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model
model_with_aug = make_model()

aug_history = model_with_aug.fit(augmented_train_batches, epochs=50, validation_data=validation_batches)