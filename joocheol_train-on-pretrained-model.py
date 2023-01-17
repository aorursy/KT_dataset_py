# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%%time



from __future__ import absolute_import, division, print_function



import tensorflow as tf

import IPython.display as display



tf.enable_eager_execution()

print(tf.VERSION)



AUTOTUNE = tf.data.experimental.AUTOTUNE
%%time



img_size = 10



df = pd.read_csv('../input/iwildcam-2019-fgvc6/train.csv')



f = df['file_name']

id = df['category_id']



all_image_paths = ['../input/iwildcam-2019-fgvc6/train_images/' + fname for fname in f]

all_image_labels = [i for i in id]



paths_labels = dict(zip(all_image_paths[0:img_size], all_image_labels[0:img_size]))
%%time



mobile_net = tf.keras.applications.DenseNet121(weights='imagenet', input_shape=(192, 192, 3), include_top=False)

mobile_net.trainable=False
%%time



ds = tf.data.TFRecordDataset('../input/let-s-make-tfrecord-simple-version/images.tfrec')



def parse(x):

  result = tf.io.parse_tensor(x, out_type=tf.string)

  result = tf.image.decode_jpeg(result, channels=3)

  result = tf.dtypes.cast(result, tf.float32)

#  result = 2 * (result/255.) - 1

  result = result/255.

#  result = tf.reshape(result, [28, 28, 3])

  return result



ds = ds.map(parse, num_parallel_calls=AUTOTUNE)

#ds = ds.map(change_range)

lables = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))



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
#%%time



#model.fit(ds, epochs=10, steps_per_epoch=5000)
model.fit(ds, epochs=5, steps_per_epoch=60)
!ls -al

checkpoint_path = "cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# include the epoch in the file name. (uses `str.format`)



cp_callback = tf.keras.callbacks.ModelCheckpoint(

    checkpoint_path, verbose=1, save_weights_only=True,

    period=1)



model.save_weights(checkpoint_path.format(epoch=0))

model.fit(ds, epochs = 5, steps_per_epoch = 60, callbacks = [cp_callback],

          verbose=1)
!ls -al
latest = tf.train.latest_checkpoint(checkpoint_dir)

latest
model.load_weights(latest)
model.fit(ds, epochs = 5, steps_per_epoch = 60, callbacks = [cp_callback],

          verbose=1)
!ls -al