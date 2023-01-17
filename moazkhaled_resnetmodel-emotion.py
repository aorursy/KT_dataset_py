import tflearn

import numpy as np

from numpy import genfromtxt

from tflearn.data_preprocessing import ImagePreprocessing

import pandas as pd

import cv2

n=5

X_All = np.asarray(genfromtxt('../input/gpemotion/fer2013.csv', delimiter=' ',  skip_header=1, dtype=float))

Y_All = np.asarray(genfromtxt('../input/gpemotion/fer2013_y.csv', delimiter=',',  skip_header=1, dtype=int))

#from google.colab import files



#uploaded = files.upload()



#X_All = np.asarray(genfromtxt('fer2013.csv', delimiter=' ',  skip_header=1, dtype=float))

#Y_All = np.asarray(genfromtxt('fer2013_y.csv', delimiter=',',  skip_header=1, dtype=int))



# Reshape the images into 48x4

X_All = X_All.reshape([-1, 48, 48, 1])



# 80% of the data for trainning and it's labels

X =  X_All[:int(35887 *.8)]

print(X.shape)



Y =  Y_All[:int(35887 *.8)]

print(Y.shape)





# 20% of the data for testing and it's labels

X_test = X_All[int(35887 *.8):]

print(X_test.shape)



Y_test = Y_All[int(35887 *.8):]

print(Y_test.shape)





# One hot encode the labels

Y = tflearn.data_utils.to_categorical(Y, 7)

Y_test = tflearn.data_utils.to_categorical(Y_test, 7)

# Real-time preprocessing of the image data

img_prep = ImagePreprocessing()

img_prep.add_featurewise_zero_center()

img_prep.add_featurewise_stdnorm()



# Real-time data augmentation

img_aug = tflearn.ImageAugmentation()

img_aug.add_random_flip_leftright()

# Building Residual Network

net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)

net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)

net = tflearn.residual_block(net, n, 16)

net = tflearn.residual_block(net, 1, 32, downsample=True)

net = tflearn.residual_block(net, n-1, 32)

net = tflearn.residual_block(net, 1, 64, downsample=True)

net = tflearn.residual_block(net, n-1, 64)

net = tflearn.batch_normalization(net)

net = tflearn.activation(net, 'relu')

net = tflearn.global_avg_pool(net)
# Regression

net = tflearn.fully_connected(net, 7, activation='softmax')

mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)

net = tflearn.regression(net, optimizer=mom,

                         loss='categorical_crossentropy')
# Training

model = tflearn.DNN(net , checkpoint_path='../input/models/model_resnet_emotion' , max_checkpoints=20, tensorboard_verbose=0,clip_gradients=0.)



model.load('../input/current_model/model_resnet_emotion-42000',weights_only=True)
model.fit(X, Y, n_epoch=150, snapshot_epoch=False, snapshot_step=500,

          show_metric=True, batch_size=128, shuffle=True, run_id='resnet_emotion')



score = model.evaluate(X_test, Y_test)

print('Test accuarcy: ', score)