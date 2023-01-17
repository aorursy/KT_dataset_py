#Any results This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import IPython.display as display

import sys



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output. you write to the current directory are saved as output.
df = pd.read_csv('../input/train-csv/train.csv')

f = df['filename']

id = df['class']

all_image_labels = [i for i in id]

#all_image_labels

%%time

mobile_net = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,input_shape=(192, 192, 3), weights='imagenet')

mobile_net.trainable=False
ds = tf.data.TFRecordDataset('../input/TF_records/test.record')

"""def parse(x):

  result = tf.io.parse_tensor(x, out_type=tf.string)

  tf.print(result, output_stream=sys.stdout)

  print(result)

  result = tf.image.decode_jpeg(result, channels=3)

  print(result)

  result = tf.dtypes.cast(result, tf.float32)

  print(result)

#  result = 2 * (result/255.) - 1

  result = result/255.

  print(result)

  #result = tf.reshape(result, [28, 28, 3])

  return result



ds = ds.map(parse, num_parallel_calls= tf.data.experimental.AUTOTUNE)"""
lables = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels,tf.string))

print(lables)

ds = tf.data.Dataset.zip((ds, lables))
ds = ds.repeat()

ds = ds.batch(32)
model = tf.keras.Sequential([

  mobile_net,

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(1024, activation='relu'),

  tf.keras.layers.Dense(23, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 

              loss=tf.keras.losses.sparse_categorical_crossentropy,

              metrics=["accuracy"])
model.fit(ds, epochs=5, steps_per_epoch=60)
checkpoint_path = "cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(

    checkpoint_path, verbose=1, save_weights_only=True,

    period=1)



model.save_weights(checkpoint_path.format(epoch=0))

model.fit(ds, epochs = 5, steps_per_epoch = 60, callbacks = [cp_callback],

          verbose=1)
latest = tf.train.latest_checkpoint(checkpoint_dir)

latest
model.load_weights(latest)

model.fit(ds, epochs = 5, steps_per_epoch = 60, callbacks = [cp_callback],

          verbose=1)