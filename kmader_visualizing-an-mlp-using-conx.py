!pip install -q conx
%matplotlib inline
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
try:
    from skimage.util.montage import montage2d
except ImportError as e:
    print('scikit-image is too new, ',e)
    from skimage.util import montage as montage2d
DATA_ROOT_PATH = os.path.join('..', 'input')
with np.load(os.path.join(DATA_ROOT_PATH, 'train.npz')) as npz_data:
    train_img = np.expand_dims(npz_data['img'], -1)
    train_idx = npz_data['idx'] # the id for each image so we can match the labels
    print('image shape', train_img.shape)
    print('idx shape', train_idx.shape)
train_labels = pd.read_csv(os.path.join(DATA_ROOT_PATH, 'train_labels.csv'))
train_dict = dict(zip(train_labels['idx'], train_labels['label'])) # map idx to label
train_labels.head(4)
import conx as cx
cx.dynamic_pictures();
net = cx.Network('MNIST_MLP')

net.add(cx.Layer('input', (40, 40, 1)))
net.add(cx.FlattenLayer('flat_input'))
net.add(cx.Layer('hidden1', 512, activation='relu'))
net.add(cx.Layer('hidden2', 32, activation='relu'))
net.add(cx.Layer('output', 10, activation='softmax'))

# creates connections between layers in the order they were added
net.connect()
net.compile(error='categorical_crossentropy',
            optimizer='sgd', lr=1e-3, decay=1e-6)
net.picture()
# add the affmnist data
from keras.utils.np_utils import to_categorical 
net.dataset.clear()
xy_pairs = [(x, to_categorical(train_dict[y], 10)) 
            for i, (x, y) in enumerate(zip(train_img, train_idx))
           if i<50000]
print('adding', len(xy_pairs), 'to output')
net.dataset.append(xy_pairs)
net.dataset.split(0.25)
net.train(epochs=5, record=True, batch_size=1024)
net.dashboard()
net.picture(train_img[1], 
            dynamic = True, 
            rotate = True, 
            show_targets = True, 
            scale = 1.25)
net.picture(train_img[2], 
            dynamic = True, 
            rotate = True, 
            show_targets = True, 
            scale = 1.25)
net.movie(lambda net, epoch: net.propagate_to_image("hidden1", train_img[2], scale = 15), 
                'early_hidden.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("hidden2", train_img[2], scale = 15), 
                'early_hidden.gif', mp4 = False)
net.train(epochs=50, record=True)
