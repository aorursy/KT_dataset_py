# !pip install tensorflow-gpu==2.0.0-beta1
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import pathlib

import tensorflow as tf

tf.enable_eager_execution()

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

%matplotlib inline
# Compatibility operations

from tensorflow.compat.v1 import ConfigProto

from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
print('Version: {}'.format(tf.VERSION))
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Organizing paths of data locations

main_path = pathlib.Path(r"../input/oct2017/OCT2017 ")

train_path = main_path / 'train'

test_path = main_path / 'test'

val_path = main_path / 'val'



# TRAIN_PATH='../input/oct2017/OCT2017 /train'

# TEST_PATH='../input/oct2017/OCT2017 /test'

# VAL_PATH='../input/oct2017/OCT2017 /val'



train_path
import random

train_image_paths = [str(path) for path in list(train_path.glob('*/*.jpeg'))]

random.shuffle(train_image_paths)

test_image_paths = [str(path) for path in list(test_path.glob('*/*.jpeg'))]

val_image_paths = [str(path) for path in list(val_path.glob('*/*.jpeg'))]





print('Number of training images:', len(train_image_paths))

print('Number of testing images:', len(test_image_paths))

print('Number of validation images:', len(val_image_paths))
label_names = sorted(set(item.name for item in train_path.glob('*') if item.is_dir()))

label_to_index = dict((name, index) for index,name in enumerate(label_names))

label_to_index
train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]

test_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_image_paths]

val_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in val_image_paths]



print("First 10 labels indices: ", train_image_labels[:10])
ex_im = tf.read_file(train_image_paths[0])

ex_im = tf.image.decode_jpeg(ex_im, channels=1)

ex_im = tf.image.resize_images(ex_im, [192, 192])



plt.imshow(ex_im[:, :, 0])
target_im_size = [192, 192]



def preprocess_image(image):

    image = tf.image.decode_jpeg(image, channels=1)

    image = tf.image.resize_image_with_crop_or_pad(image, 496, 496) # First crop center square of image (some have extra left/right pixels)

    image = tf.image.resize_images(image, target_im_size) # Resize to final dimensions

    image /= 255.0  

    return image



def load_and_preprocess_image(path):

    image = tf.read_file(path)

    return preprocess_image(image)
# Path datasets

train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)

test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)

val_path_ds = tf.data.Dataset.from_tensor_slices(val_image_paths)





# Image datasets

train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

val_image_ds = val_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)



# Label datasets

train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_image_labels, tf.int64))

test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_image_labels, tf.int64))

val_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_image_labels, tf.int64))





# Datasets with both images and labels

train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))

test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))

val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
print('image shape: ', train_image_label_ds.output_shapes[0])

print('label shape: ', train_image_label_ds.output_shapes[1])

print('types: ', train_image_label_ds.output_types)

print()

print(train_image_label_ds)
BATCH_SIZE = 64



train_ds = train_image_label_ds.shuffle(buffer_size=400) # Shuffles datasets

train_ds = train_ds.repeat() # Creates datasets iterator

train_ds = train_ds.batch(BATCH_SIZE) # Batches dataset

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # Allows dataset to prefetch batches while training for performance



# Repeat for testing dataset

test_ds = test_image_label_ds.shuffle(buffer_size=200)

test_ds = test_ds.repeat()

test_ds = test_ds.batch(BATCH_SIZE)

test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)



# Repeat for validation dataset

val_ds = val_image_label_ds.shuffle(buffer_size=200)

val_ds = val_ds.repeat()

val_ds = val_ds.batch(BATCH_SIZE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
model = models.Sequential()



model.add(layers.Conv2D(32, (5, 5), padding='valid', activation='relu', input_shape=(*target_im_size, 1))) # CNN Layer 1

model.add(layers.MaxPooling2D((2, 2))) # Pooling layer 1



model.add(layers.Conv2D(64, (5, 5), activation='relu')) # CNN Layer 2

model.add(layers.MaxPooling2D((2, 2))) # Pooling layer 2



model.add(layers.Conv2D(128, (5, 5), activation='relu')) # CNN Layer 3

model.add(layers.Flatten()) # Flattening layer



model.add(layers.Dense(64, activation='relu')) # Fully-connected layer "on top"

model.add(layers.Dropout(0.2))



model.add(layers.Dense(4, activation='softmax')) # Softmax output from logits
model.summary()
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
import os

# Directory where the checkpoints will be saved

checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
EPOCHS = 1

model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=len(train_image_paths)//BATCH_SIZE, callbacks=[checkpoint_callback])
test_loss, test_acc = model.evaluate(test_ds, steps=len(test_image_paths))
print('Model Accuracy on Test Data: {:.1f}%'.format(test_acc * 100))