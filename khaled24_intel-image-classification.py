import os
for dirname,_,_ in os.walk('/kaggle/input'):
    print(dirname)
# Libraries
from zipfile import ZipFile
import shutil, os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard,LearningRateScheduler
import tensorflow as tf
import datetime
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64 #16 * strategy.num_replicas_in_sync
SHUFFLE_SIZE = 5000
IMAGE_SIZE = [150, 150]
EPOCHS = 30
# Load the data
train_filenames = tf.io.gfile.glob(str('/kaggle/input/intel-image-classification/seg_train/seg_train/*/*'))
val_filenames = tf.io.gfile.glob(str('/kaggle/input/intel-image-classification/seg_test/seg_test/*/*'))

print(len(train_filenames),len(val_filenames))
count_dict = {'NB_SEA': 0, 'NB_FOREST': 0, 'NB_MOUNTAIN': 0, 'NB_GLACIER': 0, 'NB_BUILDINGS': 0, 'NB_STREET': 0}
for elt in os.listdir('/kaggle/input/intel-image-classification/seg_test/seg_test'):
    for key in count_dict.keys():
        if elt.upper() in key:
            count_dict[key] = len([name for name in train_filenames if elt in name])
            
count_dict
# Creating datasets
train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

for f in train_list_ds.take(5):
    print(f.numpy())
# Training images count
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

# Validation images count
VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))
# Class names
CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in tf.io.gfile.glob(str('/kaggle/input/intel-image-classification/seg_train/seg_train/*')) if '__' not in item])
CLASS_NAMES
def get_label(file_path):
    
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def image_processing(file_path):
    
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size
    img = tf.image.resize(img, IMAGE_SIZE)
    
    return img, label
# Mapped trainset
train_ds = train_list_ds.map(image_processing, num_parallel_calls=AUTOTUNE)

# Mapped Validationset
val_ds = val_list_ds.map(image_processing, num_parallel_calls=AUTOTUNE)

# Shape and label for one image
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
# Load and format testset
test_filenames = tf.io.gfile.glob(str('/kaggle/input/intel-image-classification/seg_pred/seg_pred/*'))
test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(image_processing, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

print("Testing images count: " + str(TEST_IMAGE_COUNT))
# This is a small dataset, only load it once, and keep it in memory.
# Shuffle it and repeat forever
# Batch the dataset
# `prefetch` lets the dataset fetch batches in the background while the model is training.
train_ds = train_ds.cache().shuffle(SHUFFLE_SIZE).repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.cache().shuffle(SHUFFLE_SIZE).repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)

image_batch, label_batch = next(iter(train_ds))
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(16):
        ax = plt.subplot(4,4,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis("off")

# Visualize the first batch
show_batch(image_batch.numpy(), label_batch.numpy())
weight_for_0 = (1 / count_dict["NB_SEA"])*(TRAIN_IMG_COUNT)/6.0 
weight_for_1 = (1 / count_dict["NB_FOREST"])*(TRAIN_IMG_COUNT)/6.0
weight_for_2 = (1 / count_dict["NB_MOUNTAIN"])*(TRAIN_IMG_COUNT)/6.0 
weight_for_3 = (1 / count_dict["NB_GLACIER"])*(TRAIN_IMG_COUNT)/6.0
weight_for_4 = (1 / count_dict["NB_BUILDINGS"])*(TRAIN_IMG_COUNT)/6.0 
weight_for_5 = (1 / count_dict["NB_STREET"])*(TRAIN_IMG_COUNT)/6.0

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3, 4: weight_for_4, 5: weight_for_5}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
print('Weight for class 2: {:.2f}'.format(weight_for_2))
print('Weight for class 3: {:.2f}'.format(weight_for_3))
print('Weight for class 4: {:.2f}'.format(weight_for_4))
print('Weight for class 5: {:.2f}'.format(weight_for_5))
# Modeling
"""MODULE_HANDLE = "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4"
feature_extractor = hub.KerasLayer(MODULE_HANDLE, input_shape= (150,150) + (3,), output_shape=[1536], trainable=False)"""
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(MODULE_HANDLE, input_shape= (150,150) + (3,), output_shape=[2048], trainable=False) # 2048
model = Sequential([feature_extractor, Dense(6,activation='softmax')])

# Learning rate decay / scheduling
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
rmsprop = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop")

# Metrics
METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

# Model Compile
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=METRICS)

# Callbacks functions
logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
rLRop = ReduceLROnPlateau(monitor="val_accuracy", factor=0.8, patience=3, verbose=1, mode="auto", min_lr=0.000001)
tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=False, update_freq="batch", profile_batch=0,
                          embeddings_freq=0,embeddings_metadata=None)
# Model fit function
history = model.fit(
    train_ds,
    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
    #class_weight=class_weight,
    callbacks=[tensorboard,rLRop]
)
# Visualize model's performance
fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
# Predictions
loss, acc, prec, rec = model.evaluate(val_ds, steps=VAL_IMG_COUNT // BATCH_SIZE)
