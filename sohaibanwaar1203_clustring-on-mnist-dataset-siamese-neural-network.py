# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

from tensorflow.examples.tutorials.mnist import input_data

# Any results you write to the current directory are saved as output.
# mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

mnist = input_data.read_data_sets("../input/", one_hot = False, validation_size=0)

X_train=mnist.train.images

y_train=mnist.train.labels

X_test=mnist.test.images

y_test=mnist.test.labels





X_train=X_train.reshape(X_train.shape[0],28,28,1)

X_test=X_test.reshape(X_test.shape[0],28,28,1)

print("Shape of X_train",X_train.shape)

print("Shape of Y_train",y_train.shape)

print("Shape of X_test",X_test.shape)

print("Shape of Y_test",y_test.shape)
plt.imshow(X_train[3][:,:,0])
images_list=[]

f, axarr = plt.subplots(5,4, figsize=(10,10))

#you can select you images range from here image range should be twenty in between o to 19000

begning_of_images=1000

ending_of_images=1020



count=0

for i in range(5):

    for j in range(4):

        

            axarr[i,j].title.set_text(y_train[count])

            axarr[i,j].imshow(X_train[count][:,:,0])

            count=count+1
train_groups = [X_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]

test_groups = [X_test[np.where(y_test==i)[0]] for i in np.unique(y_test)]

print('train groups:', [X.shape[0] for X in train_groups])

print('test groups:', [X.shape[0] for X in test_groups])
# Import Keras and other Deep Learning dependencies

from keras.models import Sequential

import time

from keras.optimizers import Adam

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

import seaborn as sns

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.initializers import glorot_uniform

from sklearn.preprocessing import LabelBinarizer

from keras.optimizers import *

from keras.engine.topology import Layer

from keras import backend as K

from keras.regularizers import l2

K.set_image_data_format('channels_last')

import cv2

import os

from skimage import io

import numpy as np

from numpy import genfromtxt

import pandas as pd

import tensorflow as tf



import numpy.random as rng

from sklearn.utils import shuffle



%matplotlib inline

%load_ext autoreload

%reload_ext autoreload

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

def gen_random_batch(in_groups, batch_halfsize = 8):

    out_img_a, out_img_b, out_score = [], [], []

    all_groups = list(range(len(in_groups)))

    for match_group in [True, False]:

        group_idx = np.random.choice(all_groups, size = batch_halfsize)

        out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]

        if match_group:

            b_group_idx = group_idx

            out_score += [1]*batch_halfsize

        else:

            # anything but the same group

            non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 

            b_group_idx = non_group_idx

            out_score += [0]*batch_halfsize

            

        out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

            

    return np.stack(out_img_a,0), np.stack(out_img_b,0), np.stack(out_score,0)


pv_a, pv_b, pv_sim = gen_random_batch(train_groups, 3)

fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))

for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):

    ax1.imshow(c_a[:,:,0])

    ax1.set_title('Image A')

    ax1.axis('off')

    ax2.imshow(c_b[:,:,0])

    ax2.set_title('Image B\n Similarity: %3.0f%%' % (100*c_d))

    ax2.axis('off')
from keras.models import Model

from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout

img_in = Input(shape = X_train.shape[1:], name = 'FeatureNet_ImageInput')

n_layer = img_in

for i in range(2):

    n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)

    n_layer = BatchNormalization()(n_layer)

    n_layer = Activation('relu')(n_layer)

    n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)

    n_layer = BatchNormalization()(n_layer)

    n_layer = Activation('relu')(n_layer)

    n_layer = MaxPool2D((2,2))(n_layer)

n_layer = Flatten()(n_layer)

n_layer = Dense(32, activation = 'linear')(n_layer)

n_layer = Dropout(0.5)(n_layer)

n_layer = BatchNormalization()(n_layer)

n_layer = Activation('relu')(n_layer)

feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')

feature_model.summary()
from keras.layers import concatenate

img_a_in = Input(shape = X_train.shape[1:], name = 'ImageA_Input')

img_b_in = Input(shape = X_train.shape[1:], name = 'ImageB_Input')

img_a_feat = feature_model(img_a_in)

img_b_feat = feature_model(img_b_in)

combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')

combined_features = Dense(16, activation = 'linear')(combined_features)

combined_features = BatchNormalization()(combined_features)

combined_features = Activation('relu')(combined_features)

combined_features = Dense(4, activation = 'linear')(combined_features)

combined_features = BatchNormalization()(combined_features)

combined_features = Activation('relu')(combined_features)

combined_features = Dense(1, activation = 'sigmoid')(combined_features)

similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'Similarity_Model')

similarity_model.summary()

    
# setup the optimization process

similarity_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['mae'])

def show_model_output(nb_examples = 3):

    pv_a, pv_b, pv_sim = gen_random_batch(test_groups, nb_examples)

    pred_sim = similarity_model.predict([pv_a, pv_b])

    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))

    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, pred_sim, m_axs.T):

        ax1.imshow(c_a[:,:,0])

        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*c_d))

        ax1.axis('off')

        ax2.imshow(c_b[:,:,0])

        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p_d))

        ax2.axis('off')

    return fig

# a completely untrained model

_ = show_model_output()
# make a generator out of the data

def siam_gen(in_groups, batch_size = 32):

    while True:

        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)

        yield [pv_a, pv_b], pv_sim

# we want a constant validation group to have a frame of reference for model performance

valid_a, valid_b, valid_sim = gen_random_batch(test_groups, 1024)

loss_history = similarity_model.fit_generator(siam_gen(train_groups), 

                               steps_per_epoch = 500,

                               validation_data=([valid_a, valid_b], valid_sim),

                                              epochs = 30,

                                             verbose = True)
_ = show_model_output()
t_shirt_vec = np.stack([train_groups[0][0]]*X_test.shape[0],0)

t_shirt_score = similarity_model.predict([t_shirt_vec, X_test], verbose = True, batch_size = 128)

ankle_boot_vec = np.stack([train_groups[-1][0]]*X_test.shape[0],0)

ankle_boot_score = similarity_model.predict([ankle_boot_vec, X_test], verbose = True, batch_size = 128)
obj_categories =["1","2","3","4","5","6","7","8","9"]

colors = plt.cm.rainbow(np.linspace(0, 1, 10))

plt.figure(figsize=(10, 10))



for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):

    plt.scatter(t_shirt_score[np.where(y_test == c_group), 0],

                ankle_boot_score[np.where(y_test == c_group), 0],

                marker='.',

                color=c_color,

                linewidth='1',

                alpha=0.8,

                label=c_label)

plt.xlabel('T-Shirt Dimension')

plt.ylabel('Ankle-Boot Dimension')

plt.title('T-Shirt and Ankle-Boot Dimension')

plt.legend(loc='best')

plt.savefig('tshirt-boot-dist.png')

plt.show(block=False)
x_test_features = feature_model.predict(X_test, verbose = True, batch_size=128)
%%time

from sklearn.manifold import TSNE

tsne_obj = TSNE(n_components=2,

                         init='pca',

                         random_state=101,

                         method='barnes_hut',

                         n_iter=500,

                         verbose=2)

tsne_features = tsne_obj.fit_transform(x_test_features)
obj_categories =["1","2","3","4","5","6","7","8","9"]

colors = plt.cm.rainbow(np.linspace(0, 1, 10))

plt.figure(figsize=(10, 10))



for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):

    plt.scatter(tsne_features[np.where(y_test == c_group), 0],

                tsne_features[np.where(y_test == c_group), 1],

                marker='o',

                color=c_color,

                linewidth='1',

                alpha=0.8,

                label=c_label)

plt.xlabel('Dimension 1')

plt.ylabel('Dimension 2')

plt.title('t-SNE on Testing Samples')

plt.legend(loc='best')

plt.savefig('clothes-dist.png')

plt.show(block=False)