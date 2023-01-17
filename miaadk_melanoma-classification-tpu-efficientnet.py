import numpy as np
import pandas as pd
import math
import random

!pip install -q efficientnet

import tensorflow as tf
tf.random.set_seed(5)
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate, Activation, LeakyReLU, ReLU, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets

import os
import re
from tqdm import tqdm
from datetime import datetime
try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
    print('TPU>>\t', TPU.master())

except ValueError:
    TPU = None
    strategy = tf.distribute.get_strategy()
    raise ValueError

print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
Gcs_Path = KaggleDatasets().get_gcs_path("siim-isic-melanoma-classification")
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
# Directories
Data_Dir = r"/kaggle/input/siim-isic-melanoma-classification/{}.csv"

# Config
Epochs = 10
Batch_Size = 8 * strategy.num_replicas_in_sync
Image_Size = (1024, 1024)
Class_Weight = {0: 1, 1: 3.32}

Num_Training_Images = None
Num_Test_Images = None

Early_Stop = EarlyStopping(monitor="val_accuracy",
                           mode="auto",
                           verbose=1,
                           patience=1,
                           restore_best_weights=True
                          )
Sample_Submission  = pd.read_csv(Data_Dir.format("sample_submission"))
Train_Labels  = pd.read_csv(Data_Dir.format("train"))
Test_Labels = pd.read_csv(Data_Dir.format("test"))

Train_Labels.head()
Training_File_Names = tf.io.gfile.glob(Gcs_Path + "/tfrecords/train*.tfrec")
Test_File_Names = tf.io.gfile.glob(Gcs_Path + "/tfrecords/test*.tfrec")
def data_augment(image, label):
    augment_mode = random.randint(1, 7)
    
    if augment_mode == 1:
        image = tf.image.random_flip_left_right(image)
    
    elif augment_mode == 2:
        image = tf.image.random_flip_up_down(image)
    
    elif augment_mode == 3:
        image = tf.image.central_crop(image, central_fraction=0.1)
    
    elif augment_mode == 4:
        image = tf.image.central_crop(image, central_fraction=0.1)
        image = tf.image.random_flip_up_down(image)
    
    elif augment_mode == 5:
        image = tf.image.rot90(image)
        image = tf.image.random_flip_left_right(image)
    
    elif augment_mode == 6 or augment_mode == 7:
        pass
    
    return image, label

        
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0    # Normalizing image
    image = tf.reshape(image, [*Image_Size, 3])
    return image                 
def read_labeled_tfrecord(example):
    Labeled_Tfrec_Format = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    
    example = tf.io.parse_single_example(example, Labeled_Tfrec_Format)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    
    return image, label


def read_unlabeled_tfrecord(example):
    Unlabeled_Tfrec_Format = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    
    example = tf.io.parse_single_example(example, Unlabeled_Tfrec_Format)
    image = decode_image(example["image"])
    image_ID = example['image_name']
    
    return image, image_ID


def load_dataset(file_names, labeled=True, ordered=False):
    options = tf.data.Options()  
    
    if not ordered:
        options.experimental_deterministic = False    # ignore order
    
    dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=AUTO)
    dataset = dataset.with_options(options)           # ignore order
    
    if labeled:
        dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
        
    else:
        dataset = dataset.map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    
    return dataset
def get_training_dataset(file_names, batch_size):
    dataset = load_dataset(file_names, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(file_names, batch_size, ordered=False):
    dataset = load_dataset(file_names, labeled=False, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(file_names):
    n = []
    
    for file_name in file_names:
        n.append(int(re.compile(r"-([0-9]*)\.").search(file_name).group(1)))

    return np.sum(n)
Num_Training_Images = count_data_items(Training_File_Names)
Num_Test_Images = count_data_items(Test_File_Names)
print(f"info:\n{Num_Training_Images} training images\t{Num_Test_Images} test images")
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn
with strategy.scope():
    base_model = efn.EfficientNetB7(include_top=False, input_shape=(*Image_Size, 3), weights="imagenet")
    
    Model = Sequential()
    
    Model.add(base_model)
    Model.add(GlobalAveragePooling2D())
    Model.add(Dense(1, activation="sigmoid"))
    
    Model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1),
        metrics=["accuracy", "binary_crossentropy"]
    )

Model.summary()
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

TB_Callback = TensorBoard(log_dir=f"/{Model.name}/", histogram_freq=1)
Model.Name = "EfficientNetB7"

history = Model.fit(get_training_dataset(Training_File_Names, Batch_Size),
                    epochs=Epochs,
                    steps_per_epoch=Num_Training_Images / Batch_Size,
                    callbacks=[lr_schedule],
                    class_weight=Class_Weight,
                    verbose=1
                   )
Model.save(Model.name + "-" + str(datetime.now())+ ".h5")
print('Testing Model...')
Test_Dataset = get_test_dataset(Test_File_Names, Batch_Size)

Test_Images_Dataset = Test_Dataset.map(lambda image, image_ID: image)   # remove IDs
Test_IDs_Dtaset = Test_Dataset.map(lambda image, image_ID: image_ID).unbatch()

Predictions = Model.predict(Test_Images_Dataset)
Test_IDs = next(iter(Test_IDs_Dtaset.batch(Num_Test_Images))).numpy().astype("U")
My_Submission = pd.DataFrame({"image_name": Test_IDs, "target": np.concatenate(Predictions)})
My_Submission.to_csv(f"submission-{Model.Name}-{str(datetime.now())}.csv", index=False)

print(Sample_Submission.head())
My_Submission.head()