!pip install -qq conx
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
data_dir = '../input/'
def categories_to_indicators(in_df):
    new_df = in_df.copy()
    new_df['IsMale'] = in_df['PatientSex'].map(lambda x: 'M' in x).astype(float)
    new_df['IsAP'] = in_df['ViewPosition'].map(lambda x: 'AP' in x).astype(float)
    return new_df.drop(['PatientSex', 'ViewPosition'], axis=1)
full_train_df = categories_to_indicators(pd.read_csv(os.path.join(data_dir, 'train_all.csv')))
full_stack = imread(os.path.join(data_dir, 'train.tif')) # read all slices
full_train_df['image'] = full_train_df['slice_idx'].map(lambda x: full_stack[x]) # get the slice
full_train_df.sample(3)
from keras.utils.np_utils import to_categorical
image_array = np.expand_dims(np.stack(full_train_df['image'].values, 0), -1)/255.0
opacity_index = np.expand_dims(np.stack(full_train_df['opacity'].values, 0), -1)
opacity_array = to_categorical(opacity_index)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
ax1.imshow(image_array[1, :, :, 0], interpolation = 'none', cmap = 'bone')
ax1.set_title(opacity_array[1])

ax2.imshow(image_array[2, :, :, 0], interpolation = 'none', cmap = 'bone')
ax2.set_title(opacity_array[2]);
import conx as cx
cx.dynamic_pictures();
net = cx.Network("PneuNet")
net.add(cx.ImageLayer("input", (64, 64), 1))
net.add(cx.BatchNormalizationLayer("batchnnorm"))
net.add(cx.Conv2DLayer("conv1", 8, (3, 3), padding='same', activation='relu'))
net.add(cx.Conv2DLayer("conv2", 8, (3, 3), activation='relu'))
net.add(cx.MaxPool2DLayer("pool1", pool_size=(2, 2), dropout=0.25))
net.add(cx.Conv2DLayer("conv3", 16, (3, 3), padding='same', activation='relu'))
net.add(cx.Conv2DLayer("conv4", 16, (3, 3), activation='relu'))
net.add(cx.MaxPool2DLayer("pool2", pool_size=(2, 2), dropout=0.25))
net.add(cx.Conv2DLayer("conv5", 32, (3, 3), padding='same', activation='relu'))
net.add(cx.Conv2DLayer("conv6", 32, (3, 3), activation='relu'))
net.add(cx.MaxPool2DLayer("pool3", pool_size=(2, 2), dropout=0.25))
net.add(cx.FlattenLayer("flatten"))
net.add(cx.Layer("hidden1", 64, activation='relu', vshape=(8, 8), dropout=0.5))
net.add(cx.Layer("output", 2, activation='softmax'))
net.connect()
net.compile(error='categorical_crossentropy',
            optimizer='rmsprop', lr=3e-3, decay=1e-6)
net.summary()
net.picture(image_array[1], 
            dynamic = True, rotate = True, show_targets = True, scale = 1.25)
net.dataset.clear()
ip_pairs = list(zip(image_array, opacity_array))
print('adding', len(ip_pairs), 'to output')
net.dataset.append(ip_pairs)
net.dataset.split(0.25)
net.train(epochs=30, record=True)
net.propagate_to_image("conv5", image_array[1])
net.picture(image_array[1], dynamic = True, rotate = True, show_targets = True, scale = 1.25)
net.picture(image_array[2], dynamic = True, rotate = True, show_targets = True, scale = 1.25)
net.dashboard()
net.movie(lambda net, epoch: net.propagate_to_image("conv2", image_array[1], scale = 3), 
                'early_conv_healthy.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("conv2", image_array[2], scale = 3), 
                'early_conv_opacity.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("conv5", image_array[1], scale = 3), 
                'mid_conv_healthy.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("conv5", image_array[2], scale = 3), 
                'mid_conv_opacity.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("pool3", image_array[1], scale = 2), 
                'hr_conv_healthy.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("pool3", image_array[2], scale = 2), 
                'hr_conv_opacity.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("hidden1", image_array[1], scale = 3), 
                'hidden_healthy.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("hidden1", image_array[2], scale = 3), 
                'hidden_opacity.gif', mp4 = False)
