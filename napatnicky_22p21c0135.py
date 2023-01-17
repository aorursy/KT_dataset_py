# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
df.head()
train,val  = train_test_split(df,test_size = 0.2,random_state = 42)
def augment(img,label):
    #img = tf.image.random_flip_up_down(img)
    #img = tf.image.random_flip_left_right(img)
    #img = tf.image.random_contrast(img, 0.2, 0.5)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
    return img,label
def read_image(file_names,label):
    img = tf.io.read_file('../input/thai-mnist-classification/train/'+file_names)
    # decode to image file 
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    label = tf.one_hot(label,10)
    #img = tf.image.per_image_standardization(img)
    return img/255,label
ds_train = tf.data.Dataset.from_tensor_slices((tf.constant(train['id']),tf.constant(train['category'])))
ds_val = tf.data.Dataset.from_tensor_slices((tf.constant(val['id']),tf.constant(val['category'])))

ds_train = ds_train.map(read_image).map(augment).batch(16)
ds_val = ds_val.map(read_image).batch(16)
tf.compat.v1.reset_default_graph()
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(256, 256, 3),
    include_top=False)
inputs = tf.keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
#x = tf.keras.layers.Dense(10,activation = 'relu')(x)
outputs = tf.keras.layers.Dense(10,activation = 'softmax')(x)

model = tf.keras.Model(inputs, outputs)
x,y = next(iter(ds_train))
print(model(x))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001)
              ,loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'
    ])
hist = model.fit(ds_train,validation_data = ds_val,epochs = 3,callbacks = [])
model.evaluate(ds_val)
file_to_number = pd.read_csv('../input/covert-filename/convert_train_mnist.csv')
file_to_number['feature1'] = file_to_number['feature1'].fillna(-1)
train_rule , test_rule = train_test_split(file_to_number,test_size = 0.2,random_state = 42)
xx = train_rule['feature1']
yy = train_rule['feature2']
zz = train_rule['feature3']
pred = train_rule['predict']
train_number = tf.data.Dataset.from_tensor_slices((xx,yy,zz))
label_number = tf.data.Dataset.from_tensor_slices(pred)
train_ds = tf.data.Dataset.zip((train_number,label_number)).batch(16)
xx = test_rule['feature1']
yy = test_rule['feature2']
zz = test_rule['feature3']
pred = test_rule['predict']
train_number = tf.data.Dataset.from_tensor_slices((xx,yy,zz))
label_number = tf.data.Dataset.from_tensor_slices(pred)
val_ds = tf.data.Dataset.zip((train_number,label_number)).batch(16)
inputs1 = tf.keras.layers.Input(shape=(1, ), name='input1')
inputs2 = tf.keras.layers.Input(shape=(1, ), name='input2')
inputs3 = tf.keras.layers.Input(shape=(1, ), name='input3')
linear_model = tf.keras.Sequential([
      #tf.keras.layers.Dense(256, activation='sigmoid'),## add
      tf.keras.layers.Dense(128, activation='sigmoid'),##128
      tf.keras.layers.Dense(64, activation='sigmoid'),##64
      tf.keras.layers.Dense(16, activation='sigmoid'),##32
      #tf.keras.layers.Dense(8, activation='sigmoid'),##add
      tf.keras.layers.Dense(1)
  ])
total = tf.keras.layers.Concatenate()([inputs1,inputs2,inputs3])
out = linear_model(total)
model = tf.keras.Model(inputs=[inputs1, inputs2,inputs3], outputs=out)
x,y = next(iter(train_ds))
print(model(x))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
              ,loss=tf.keras.losses.MeanSquaredError())

hist = model.fit(train_ds,validation_data = val_ds,epochs = 3,callbacks = [])
x = tf.constant(-1)
x = tf.reshape(x,[-1,1])
y = tf.constant(4)
y = tf.reshape(y,[-1,1])
z = tf.constant(4)
z = tf.reshape(y,[-1,1])
print(model((x,y,z)))
