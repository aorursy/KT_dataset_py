# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
# from keras.preprocessing.image import ImageDataGenerator
# from keras import callbacks
# from keras.utils.vis_utils import plot_model

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# print(os.listdir("../input"))

# # Any results you write to the current directory are saved as output.
# import keras.backend as K
# import tensorflow as tf
# from keras import initializers, layers

# class Length(layers.Layer):
#     """
#     Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
#     inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
#     output: shape=[dim_1, ..., dim_{n-1}]
#     """
#     def call(self, inputs, **kwargs):
#         return K.sqrt(K.sum(K.square(inputs), -1))

#     def compute_output_shape(self, input_shape):
#         return input_shape[:-1]
    
# class Mask(layers.Layer):
#     """
#     Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
#     Output shape: [None, d2]
#     """
#     def call(self, inputs, **kwargs):
#         # use true label to select target capsule, shape=[batch_size, num_capsule]
#         if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
#             assert len(inputs) == 2
#             inputs, mask = inputs
#         else:  # if no true label, mask by the max length of vectors of capsules
#             x = inputs
#             # Enlarge the range of values in x to make max(new_x)=1 and others < 0
#             x = (x - K.max(x, 1, True)) / K.epsilon() + 1
#             mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

#         # masked inputs, shape = [batch_size, dim_vector]
#         inputs_masked = K.batch_dot(inputs, mask, [1, 1])
#         return inputs_masked

#     def compute_output_shape(self, input_shape):
#         if type(input_shape[0]) is tuple:  # true label provided
#             return tuple([None, input_shape[0][-1]])
#         else:
#             return tuple([None, input_shape[-1]])
