
from __future__ import division, print_function, absolute_import

import json
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

f = open('../input/planesnet.json')
planesnet = json.load(f)
f.close()
# Preprocess image data and labels
X = np.array(planesnet['data']) / 255.
X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
Y = np.array(planesnet['labels'])
Y = to_categorical(Y, 2)
# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
# Convolutional network building
network = input_data(shape=[None, 20, 20, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0)
# Train the model
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=.2,show_metric=True, batch_size=128, run_id='planesnet')

import random
from matplotlib import pyplot as plt

for i in range(20):
    # Choose a random image and its label
    rand_int = random.randrange(0,len(planesnet['data']))
    img = np.array(planesnet['data'][rand_int]) / 255.
    img = img.reshape((3, 400)).T.reshape((20,20,3))
    label = planesnet['labels'][rand_int]
    
    # Display the image
    plt.imshow(img)
    plt.show()
    
    # Predict the image class
    prediction = model.predict_label([img])[0][0]
    
    # Output acutal and predicted class - 0 = 'no-plane', 1 = 'plane'
    print('Actual Class: ' + str(label))
    print('Predicted Class: ' + str(prediction))

# Any results you write to the current directory are saved as output.


