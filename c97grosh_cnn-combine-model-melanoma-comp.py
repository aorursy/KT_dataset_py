# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import Model, models, layers, optimizers, losses, Input 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from PIL import Image 
import json, os, time, io, gc 
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from img_pull_and_preprocess import imgs, names, vals

data_path = '/kaggle/input/siim-isic-melanoma-classification/'
tfrec_loc = data_path+'tfrecords/'
train_data = pd.read_csv(data_path+'train.csv')
tf_rec_files = [[file for file in files if 'train' in file] \
                for _, _, files in os.walk(tfrec_loc)][0]
# image_desc = {
#     'image': tf.io.FixedLenFeature([], tf.string), 
#     'image_name': tf.io.FixedLenFeature([], tf.string), 
#     'target': tf.io.FixedLenFeature([], tf.int64),
# }

# def resize_image(img_arr):
#     new_img = Image.fromarray(img_arr).resize(size=(224, 224))
#     ret_arr = np.array(new_img)
#     return ret_arr 

# def parse_img_func(example):
#     return tf.io.parse_single_example(example, image_desc)

# def transform_rec(tfrec):
#     dataset = tf.data.TFRecordDataset(tfrec)
# #     print(sys.getsizeof(dataset))
#     parsed_set = dataset.map(parse_img_func)
#     img_arrays = [np.array(Image.open(io.BytesIO(i['image'].numpy()))) for i in parsed_set]
#     img_arrays = np.array(list(map(resize_image, img_arrays)))
#     img_names = [str(i['image_name'].numpy())[2:-1] for i in parsed_set]
#     targets = [i['target'].numpy() for i in parsed_set]
#     return img_arrays, np.array(img_names), np.array(targets)

# start = time.time()
# img_arrays, img_names, targets = [], [], []
# for i in tqdm(range(len(tf_rec_files[:2]))):
#     imgs, names, vals = transform_rec(tfrec_loc+tf_rec_files[i])
#     img_arrays.append(imgs)
#     img_names.append(names)
#     targets.append(vals)
#     next_arrays, next_names, next_targets = transform_rec(tfrec_loc+'train01-2071.tfrec') 
#     third_arrays, third_names, third_targets = transform_rec(tfrec_loc+'train02-2071.tfrec')
# imgs, names, vals = transform_rec(tfrec_loc+tf_rec_files[0])

# p = names.argsort()
# names = names[p]
# imgs = imgs[p]
# vals = vals[p]

# train_w_imgs = train_data[train_data['image_name'].isin(names)]
# train_w_imgs = train_w_imgs.sort_values(by=['image_name'])
# imgs = tf.cast(imgs, tf.float32)
# targets = np.c_[vals]

# end = time.time()
# print(end-start)
# pp_pipeline = Pipeline([
#     ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
#     ('Scaler', preprocessing.StandardScaler())
# ])
# unqs = train_w_imgs['patient_id'].unique()

# count_dict = {'patient_id': [], 'location': [], 'count': []}
# for patient in unqs:
#     counts = train_w_imgs[train_w_imgs['patient_id'] == patient]['anatom_site_general_challenge'].value_counts()
#     for i in range(len(counts.index)):
#         count_dict['patient_id'].append(patient)
#         count_dict['location'].append(counts.index[i])
#         count_dict['count'].append(counts.values[i])
# loc_counter = pd.DataFrame(count_dict)
# del count_dict
# gc.collect()
# full_df = train_w_imgs.merge(loc_counter, left_on=['patient_id', 'anatom_site_general_challenge'], \
#                            right_on=['patient_id', 'location'], how='left')
# del train_w_imgs
# del loc_counter
# del full_df['anatom_site_general_challenge']

# cols = list(full_df.columns)
# cols.remove('diagnosis')
# cols.remove('benign_malignant')
# cols.remove('target')
# cols.remove('patient_id')
# cols.remove('image_name')

# data_x = full_df[cols]
# data_x = pd.get_dummies(data_x, columns=['sex', 'location'])
# data_x = np.c_[data_x]
# data_y = np.c_[full_df['target']]

# x_train = data_x[:1450]
# x_test = data_x[1451:]

name_train = names[:1450]
name_test = names[1451:]
train_imgs = tf.cast(imgs[:1450], tf.float32)
test_imgs = tf.cast(imgs[1451:], tf.float32)
train_vals = vals[:1450]
test_vals = vals[1451:]

print(type(name_train))
print(type(train_imgs))
print(type(test_imgs))
print(type(train_vals))

del names
del imgs
del vals 
gc.collect()
# x_train_pp = pp_pipeline.fit_transform(x_train)
# x_test_pp = pp_pipeline.transform(x_test)

# del x_train
# del x_test

# gc.collect()
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
hist = pic_mod.fit(train_imgs, train_vals, batch_size=218, epochs=10)
# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# time.sleep(1)
# # instantiating the model in the strategy scope creates the model on the TPU
# with tpu_strategy.scope():
    
#     # Input for the data from the csv 
# #     csv_input = Input(shape=10)
#     # Input for the Image data 
#     img_input = Input(shape=(224, 224, 3))
    
#     # Hidden layers for the csv input 
# #     csv_1 = layers.Dense(64, activation='relu')(csv_input)
# #     csv_2 = layers.Dense(128, activation='relu')(csv_1)
# #     csv_3 = layers.Dense(128, activation='relu')(csv_2)
    
#     # Convolutional and Pooling hidden layers for the image input 
#     cnn_1 = layers.Conv2D(25, (3,3), activation='relu')(img_input)
#     pool1 = layers.MaxPooling2D((3,3))(cnn_1)
#     cnn2 = layers.Conv2D(45, (3,3), activation='relu')(pool1)
#     pool2 = layers.MaxPooling2D((3,3))(cnn2)
# #     cnn3 = layers.Conv2D(100, (3,3), activation='relu')(pool2)
#     flat = layers.Flatten()(pool2)
#     connected = layers.Dense(45, activation='relu')(flat)
    
#     # Concatenate the output from both the csv and image hidden layers 
# #     merger = layers.concatenate([csv_3, connected])
    
#     # Accepted the merged together vectors and output 
#     output = layers.Dense(1, activation='sigmoid')(connected)
    
#     # Call and compile the model 
# #     mod = Model(inputs=[csv_input, img_input], outputs=output)
#     mod = Model(inputs=img_input, outputs=output)
#     sgd = optimizers.Adam(lr=0.005)
#     mod.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
# print("Finished")
# time.sleep(1)
# # mod.fit(train_imgs, train_vals, batch_size=145, epochs=10)
# # mod([x_train_pp[0], train_imgs[0]])
# mod(train_imgs[0])
# mod.fit(train_imgs, train_vals, batch_size=218, epochs=10)