%matplotlib inline
import matplotlib.pyplot as plt

import cv2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape

import tensorflow as tf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
modelv2 = InceptionResNetV2( input_shape = (224, 224, 3), weights = "../input/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5")
images_gray = np.load('../input/l/gray_scale.npy')

images_lab = np.load('../input/ab/ab/ab1.npy')
def get_rbg_from_lab(gray_imgs, ab_imgs, n = 10):

    imgs = np.zeros((n, 224, 224, 3))

    imgs[:, :, :, 0] = gray_imgs[0:n:]

    imgs[:, :, :, 1:] = ab_imgs[0:n:]

    

    imgs = imgs.astype("uint8")

    

    imgs_ = []

    for i in range(0, n):

        imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))



    imgs_ = np.array(imgs_)



    print(imgs_.shape)

    

    return imgs_

    
def pipe_line_img(gray_scale_imgs, batch_size = 100, preprocess_f = preprocess_input):

    imgs = np.zeros((batch_size, 224, 224, 3))

    for i in range(0, 3):

        imgs[:batch_size, :, :,i] = gray_scale_imgs[:batch_size]

    return preprocess_f(imgs)
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./folder_to_save_graph_3', histogram_freq=0, write_graph=True, write_images=True)
imgs_for_input = pipe_line_img(images_gray, batch_size = 300)
imgs_for_output = preprocess_input(get_rbg_from_lab(gray_imgs = images_gray, ab_imgs = images_lab, n = 300))
plt.imshow(imgs_for_output[17])
model_simple = Sequential()

model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 12, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu))

model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 12, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu))

model_simple.add(Conv2DTranspose(strides = 1, kernel_size = 3, filters = 12, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu))

model_simple.add(Conv2DTranspose(strides = 1, kernel_size = 3, filters = 3, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu))
model_simple.compile(optimizer = tf.keras.optimizers.Adam(epsilon = 1e-8), loss = tf.losses.mean_pairwise_squared_error)

imgs_for_s = np.zeros((300, 224, 224, 1))

imgs_for_s[:, :, :, 0] = images_gray[:300] 
prediction = model_simple.predict(imgs_for_input)
prediction.shape
model_simple.fit(imgs_for_input, imgs_for_output, epochs = 15)
model_simple.fit(imgs_for_input, imgs_for_output, epochs = 1100, batch_size = 16)
out = model_simple.predict(imgs_for_input)
plt.imshow(np.squeeze(imgs_for_input[3,:])) # Input
plt.imshow(out[3,:]) # Ouput
plt.imshow(np.squeeze(imgs_for_output[3,:])) # Expected