# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tabnet
import os,shutil,tabnet

import tensorflow as tf
(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train.shape
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
@tf.function

def transform(image,label):

    x = tf.reshape(image,[-1])

    x = tf.cast(x,tf.float32)/255

    y = tf.one_hot(label,10)

    return x,y

    
batch_size = 128



ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))

ds_train = ds_train.shuffle(60000)

ds_train = ds_train.map(transform)

ds_train = ds_train.batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))

ds_test = ds_test.map(transform)

ds_test = ds_test.batch(batch_size)
model = tabnet.TabNetClassifier(feature_columns = None,num_classes = 10,num_features = 784,feature_dim = 16,output_dim = 16,

                                num_decision_steps = 1.5,relaxation_factor = 1.5,batch_momentum = 0.98,norm_type = 'group',num_groups = -1)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps = 500,decay_rate = 0.9)

optimizer = tf.keras.optimizers.Adam(lr)

model.compile(optimizer,loss = 'categorical_crossentropy',metrics = ['accuracy'])

# model.fit(ds_train,epochs = 5,validation_data = ds_test,verbose = 1)