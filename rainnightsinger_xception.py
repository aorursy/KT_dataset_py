import tensorflow as tf
print(format(tf.__version__))
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pathlib

import os

import datetime

from pathlib import Path

import zipfile
keras = tf.keras

layers =tf.keras.layers

os.listdir('../input/thyroid-trans/Picosmos_tran')
data_dir= '../input/thyroid-trans/Picosmos_tran'
data_root=pathlib.Path(data_dir)
for item in data_root.iterdir():

     print(item)
all_image_path = list(data_root.glob('*/*')) 
len(all_image_path)
all_image_path[-3:]
all_image_path = [str(path) for path in all_image_path]
all_image_path[10:12]
import random
random.shuffle(all_image_path)
all_image_path[10:12]
all_image_path=all_image_path[0:418]
image_count=len(all_image_path)

image_count
label_names = sorted(item.name for item in data_root.glob('*/'))
label_names
label_to_index =dict((name,index) for index,name in enumerate(label_names))#获取编码
all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]
all_image_label[:5]
len(all_image_label)
all_image_path[:5]
import IPython.display as display
index_to_label =dict((v,k) for k,v in label_to_index.items()) 
index_to_label
def load_preprosess_image(img_path):

    img_raw = tf.io.read_file(img_path)

    img_tensor = tf.image.decode_jpeg(img_raw,channels = 3 )

    img_tensor = tf.image.resize(img_tensor,[256,256])

    img_tensor = tf.cast(img_tensor,tf.float32)

    img = img_tensor/255

    return img#图片预处理函数
image_path = all_image_path[200]

plt.imshow(load_preprosess_image(image_path))
path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
image_dataset = path_ds.map(load_preprosess_image)
label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
for label in label_dataset.take(10):

    print(label.numpy())
for img in image_dataset.take(1):

    print(img)
dataset = tf.data.Dataset.zip((image_dataset,label_dataset))
test_count = int(image_count*0.2)

train_count = image_count-test_count
train_dataset = dataset.skip(test_count)
test_dataset = dataset.take(test_count)
BATCH_SIZE = 32
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
train_dataset
steps_per_epoch = train_count//BATCH_SIZE

validation_steps = test_count//BATCH_SIZE
new_covn_base =  keras.applications.xception.Xception(weights =None,

                                                      include_top = False,

                                                     input_shape = (256,256,3),

                                                     pooling = 'avg')
new_covn_base.trainable = True
new_model = tf.keras.Sequential([

        new_covn_base,

        layers.Dense(1024,activation='relu'),

        layers.BatchNormalization(),

        layers.Dense(512,activation='relu'),

        layers.BatchNormalization(),

        layers.Dense(3,activation='softmax')

])
new_model.compile(    optimizer='adam',#keras.optimizers.Adam(lr = 0.0001),

                      loss='sparse_categorical_crossentropy',#数组编码[1],[0],

                      #loss='categorical_crossentropy',#独热编码

                      #loss='binary_crossentropy',#二分类

                      metrics = ['acc'])
new_model.summary()
new_history = new_model.fit(

    train_dataset,

    steps_per_epoch=train_count//BATCH_SIZE,

    epochs=30,

    validation_data= test_dataset,

    validation_steps= test_count//BATCH_SIZE,

    )
plt.plot(new_history.epoch,new_history.history.get('acc'),label='acc')

plt.plot(new_history.epoch,new_history.history.get('val_acc'),label='val_acc')

plt.legend()