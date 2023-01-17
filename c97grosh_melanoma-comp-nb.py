# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from tensorflow.keras import layers, models, losses, optimizers
import tensorflow as tf 
# import xgboost as xgb 
from sklearn import preprocessing, ensemble, linear_model 
import os, sys, pickle 
from pylab import *
# import lightgbm as lgb 
# from sklearn.metrics import roc_curve, auc, accuracy_score
# from sklearn.pipeline import Pipeline 
# from sklearn.impute import SimpleImputer 
# from sklearn.model_selection import train_test_split 
import time, json, io, gc 
from PIL import Image 

def resize_image(img_arr):
    new_img = Image.fromarray(img_arr).resize(size=(224, 224))
    ret_arr = np.array(new_img)
    return ret_arr 

def jpg_img_resize(path):
    img = Image.open(path).resize(size=(224, 224))
    arr = np.array(img)
    return arr


data_path = '/kaggle/input/siim-isic-melanoma-classification/'
tfrec_loc = data_path+'tfrecords/'
# dicom_loc = data_path+'train/'

# train_data = pd.read_csv(data_path+'train.csv')
# x = os.stat(data_path+'train.csv').st_size/1e6
# print(x)
tf_rec_files = [[file for file in files if 'train' in file] \
                for _, _, files in os.walk(tfrec_loc)][0]
# count = 0
# for file in tf_rec_files:
#     count += os.stat(tfrec_loc+file).st_size/1e6

# print(count/1000)
# print(os.stat(tfrec_loc+tf_rec_files[0]).st_size/1e6)
# tf_dataset1 = tf.data.TFRecordDataset(data_path+'tfrecords/train00-2071.tfrec')
image_desc = {
    'image': tf.io.FixedLenFeature([], tf.string), 
    'image_name': tf.io.FixedLenFeature([], tf.string), 
    'target': tf.io.FixedLenFeature([], tf.int64),
}

def parse_img_func(example):
    return tf.io.parse_single_example(example, image_desc)

def transform_rec(tfrec):
    dataset = tf.data.TFRecordDataset(tfrec_loc+tfrec)
#     print(sys.getsizeof(dataset))
    parsed_set = dataset.map(parse_img_func)
    img_arrays = [np.array(Image.open(io.BytesIO(i['image'].numpy()))) for i in parsed_set]
    img_arrays = np.array(list(map(resize_image, img_arrays)))
    img_names = [i['image_name'].numpy() for i in parsed_set]
    targets = [i['target'].numpy() for i in parsed_set]
    return img_arrays, img_names, targets
start = time.time()
img_arrays, img_names, targets = transform_rec(tf_rec_files[0])
next_arrays, next_names, next_targets = transform_rec(tf_rec_files[1]) 
third_arrays, third_names, third_targets = transform_rec(tf_rec_files[2])
end = time.time()
print(end-start)
# next_arrays, next_names, next_targets = transform_rec(tf_rec_files[1]) 
comb_img_arrays = np.concatenate((img_arrays, next_arrays, third_arrays))
comb_img_names = np.concatenate((img_names, next_names, third_names))
comb_target_arrays = np.concatenate((targets, next_targets, third_targets))

targets = np.c_[comb_target_arrays]
imgs = tf.cast(comb_img_arrays, tf.float32)
# data_dict = {'imgs': comb_img_arrays, 
#             'img_names': comb_img_names, 
#             'targets': comb_target_arrays}

# data_file = open("data_pick_2.pkl", "wb")
# pickle.dump(data_dict, data_file)
# data_file.close()

# del img_arrays
# del next_arrays
# del next_names 
# del next_targets 
# del comb_img_arrays
# del comb_target_arrays
# gc.collect()

comb_img_arrays[0]
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    print("Starting up")
    pic_mod = models.Sequential([
        layers.Conv2D(96, (3,3), activation='relu', input_shape=(224, 224, 3)), 
        layers.MaxPooling2D((3,3)), 
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((3,3)), 
        layers.Conv2D(384, (3,3), activation='relu'),
    #     layers.Conv2D(384, (3,3), activation='relu'),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.Flatten(), 
        layers.Dense(128, activation='relu'), 
        layers.Dense(1, activation='sigmoid')
    ])
    sgd = optimizers.Adam(lr=0.5)
    pic_mod.compile(optimizer=sgd, loss='binary_crossentropy', 
                    metrics=[tf.keras.metrics.AUC()])
    print("Ready to train")
    time.sleep(4)
hist = pic_mod.fit(imgs, targets, batch_size=218, epochs=10)

