# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from tensorflow import keras
import os
PATH = "../input/lego-brick-images/dataset/"
first = mpl.image.imread(PATH + "2357 brick corner 1x2x2 000L.png")
print(first.dtype)
print(first.shape)
mpl.pyplot.imshow(first)
mpl.pyplot.show()
#This function extracts filenames
def load_names(directory):
    f = []
    for (filenames) in os.walk(directory):
        f.extend(filenames)
        break
    return f
data = pd.DataFrame(load_names(PATH) [2])

#Let's check:
print(data.shape)
print(data)
#Rename the first column for clarity
data.rename(columns = {0:"file_name"}, inplace=True)
#Extract the piece id at the beginning of the filename
data["piece_id"] = data["file_name"].str.split(" ").str[0]
#Drop the filename before drop duplicates
data = data.drop(["file_name"], axis=1)
#Drop duplicates
data = data.drop_duplicates()
#Reset ids
data = data.reset_index(drop=True)
#We convert the string id to numeric id
data["piece_id"] = pd.to_numeric(data["piece_id"])

#Let's check the size
print(data.shape)
print(data)
#Dictionnary of piece_id's with the order id that will be used in the classifier
dic = data.to_dict("dict")["piece_id"]

#Set two useful id vectors
conversion_vector = [i for i in range(len(dic))]
print(conversion_vector)
conversion_empty = [0 for i in range(len(dic))]
print(conversion_empty)

#General variables
CLASS_NAMES = data["piece_id"]
NB_CLASSES = len(CLASS_NAMES)
np.random.seed(42)
tf.random.set_seed(42)
#Load file names, all in the dataset directory, we remove auto shuffle
list_ds = tf.data.Dataset.list_files(str(PATH + '*'), shuffle=False)

#We shuffle the list, but not at each iteration because as we don't want train, test and validation datasets to overlap on a new run
list_ds = list_ds.shuffle(40000, seed=42, reshuffle_each_iteration=False)

#We check 2x list_ds is always shuffled in the same order
for i in list_ds.take(5):
  print(i.numpy())
for i in list_ds.take(5):
  print(i.numpy())
#Split
train_size = int(32000)
val_size = int(3200)
test_size = int(4800)

train_ds = list_ds.take(train_size)
test_ds = list_ds.skip(train_size)
val_ds = test_ds.skip(test_size)
test_ds = test_ds.take(test_size)

#Check dataset size
print(tf.data.experimental.cardinality(train_ds))
print(tf.data.experimental.cardinality(test_ds))
print(tf.data.experimental.cardinality(val_ds))
#Load and resize images to tensor with floats in the [0,1] range, at a specific size for the CNN
def decode_img(img):
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [224, 224])

#Get the label corresponding to the image returning a number in the [0,50] range , corresponding to the id in the earlier DataFrame
def get_label(file_path):
  parts=tf.strings.split(file_path,
                 os.path.sep)[-1]
  parts=tf.strings.split(parts, " ")[0]
  parts=tf.strings.to_number(parts, out_type=tf.dtypes.int64) 
  parts=tf.where(parts==CLASS_NAMES, conversion_vector, conversion_empty)
  return tf.math.reduce_sum(parts)

#Return Images and their corresponding label (id)
def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
#Create Loading using parallel calls, hypermeters hasn't been tweaked
train_load = train_ds.map(process_path, num_parallel_calls=50)
test_load = test_ds.map(process_path, num_parallel_calls=50)
val_load = val_ds.map(process_path, num_parallel_calls=50)

#Check 5 images in the train dataset
for image, label in train_load.take(5):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
"""Re-shuffle and prepare for training"""
#Cache parameter was originally set on True
def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
  # Repeat forever after shuffling at each iteration, as the sets are now clearly distinct to have different batches at each epoch
  ds = ds.repeat()
  
  #Batch by 8, this was the highest value possible with my GPU/CPU
  ds = ds.batch(8)
  
  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training. Bufer size is an hyperparameter that hasn't been tweaked, set to 1.
  ds = ds.prefetch(buffer_size=1)
  return ds
train_set = prepare_for_training(train_load)
val_set = prepare_for_training(val_load)
test_set = prepare_for_training(test_load)
#Import a Xception Model
base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
#Create a Average Pooling after the CNN layers
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)

#Create a top Dense Layer
output = keras.layers.Dense(NB_CLASSES, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

#View the CNN layers
for index, layer in enumerate(base_model.layers):
    print(index, layer.name)
#Train all layers in the model including pre-trained
for layer in base_model.layers:
    layer.trainable = True

#We set the optimizer to a Nesterov, good convergence quality, I haven't tested Nadam or RMSProp, the training is slow with the input size
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)

#We use accuracy and crossentropy to have distinct classes
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
#Steps = number of batches per epoch
history = model.fit(train_set,
                    steps_per_epoch=int(32000/8),
                    validation_data=val_set,
                    validation_steps=int(3200/8),
                    epochs=5)
model.evaluate(test_set,steps=int(4800/8))
model.save("lego_CNN_95.h5")