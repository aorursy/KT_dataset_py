import os
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

validation_labels = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_validation_v2.csv',
                        index_col='FILENAME')
validation_labels.head()

from keras.preprocessing.image import load_img
img = load_img('/kaggle/input/handwriting-recognition/validation_v2/validation/VALIDATION_0005.jpg')
img

data_dir = pathlib.Path('/kaggle/input/handwriting-recognition/validation_v2/validation/')
image_count = len(list(data_dir.glob('*.jpg')))
image_count
list_ds = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
for f in list_ds.take(5):
    print(f.numpy())
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-1]
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])
def process_path(file_path):
  #file_path='/kaggle/input/handwriting-recognition/validation_v2/validation/VALIDATION_0001'
  label = get_label(file_path)
  print(label)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
batch_size = 32
img_height = 50
img_width = 300
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
def get_label(data):
    return validation_labels.loc[data.numpy().decode('utf-8')].values[0]
for image, label in train_ds.take(2):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", get_label(label))
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds
train_ds = configure_for_performance(train_ds)

image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(get_label(label))
  plt.axis("off")
