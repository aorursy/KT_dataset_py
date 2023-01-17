import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

#read the list of train, val, trainval in Segmentation folder
files=collections.defaultdict(list)
for split in ["train", "val", "trainval"]:
    path = pjoin("/kaggle/input/pascal-voc-2012/VOC2012/ImageSets/Segmentation/", split + ".txt")
    file_list = tuple(open(path, "r"))
    file_list = [id_.rstrip() for id_ in file_list]
    files[split] = file_list
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

# from keras.models import Model, load_model
# from keras.layers import Input
# from keras.layers.core import Dropout, Lambda
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import backend as K
import torchvision.transforms as transforms

import tensorflow as tf


import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
def get_pascal_labels():
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

IMG_HEIGHT, IMG_WIDTH = 128,128
n_classes = 21

Y_train = np.zeros((len(files["train"]), IMG_HEIGHT, IMG_WIDTH, 21), dtype=int)
X_train = np.zeros((len(files["train"]), IMG_HEIGHT, IMG_WIDTH, 3), dtype=int)
pos = 0;
for ii in tqdm(files["train"]):
    fname = ii + ".png"
    Jname = ii + ".jpg"
    #label file
    lbl_path = pjoin('../input/pascal-voc-2012/VOC2012/', "SegmentationClass", fname)
    lbl_img = Image.open(lbl_path).convert('RGB')
    #resize for all the image
    lbl_img = lbl_img.resize((IMG_HEIGHT, IMG_WIDTH))
    lbl_img = np.array(lbl_img)
    label_mask = np.zeros((lbl_img.shape[0], lbl_img.shape[1]), dtype=np.int16)
    label_one_hot = np.zeros(n_classes)
    for ii_i, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(lbl_img == label, axis=-1))[:2]] = ii_i
    #one_hot_encode
    encode_label_mask = np.zeros((lbl_img.shape[0], lbl_img.shape[1],n_classes), dtype=np.int16)
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            lbl = label_mask[i][j]
            encode_label_mask[i][j][lbl] = 1
    
#     label_mask = label_mask.astype(int).reshape(IMG_HEIGHT, IMG_WIDTH,1)
#     Y_train[pos] = label_mask
    
    Y_train[pos] = encode_label_mask
    # X file
    
    jpg_path = pjoin('../input/pascal-voc-2012/VOC2012/', "JPEGImages", Jname)
    jpg_img = Image.open(jpg_path).convert('RGB')
    jpg_img = jpg_img.resize((IMG_HEIGHT, IMG_WIDTH))
    jpg_img = np.array(jpg_img)
    X_train[pos] = jpg_img
    pos = pos+1
print (Y_train[0].shape)
# X_test = np.zeros((len(files["train"]), IMG_HEIGHT, IMG_WIDTH, 3), dtype=int)
# pos = 0
# for ii in tqdm(files["val"]):
#     fname = ii + ".png"
#     Jname = ii + ".jpg"
#     jpg_path = pjoin('../input/pascal-voc-2012/VOC2012/', "JPEGImages", Jname)
#     jpg_img = Image.open(jpg_path).convert('RGB')
#     jpg_img = jpg_img.resize((IMG_HEIGHT, IMG_WIDTH))
#     jpg_img = np.array(jpg_img)
#     X_test[pos] = jpg_img
    
#     lbl_path = pjoin('../input/pascal-voc-2012/VOC2012/', "SegmentationClass", fname)
#     lbl_img = Image.open(lbl_path).convert('RGB')
#     #resize for all the image
#     lbl_img = lbl_img.resize((IMG_HEIGHT, IMG_WIDTH))
#     lbl_img = np.array(lbl_img)
#     pos = pos+1
    
# plt.imshow(X_train[10])
# plt.show()

# label_colours = get_pascal_labels()

# mask = Y_train[10].reshape(IMG_HEIGHT, IMG_WIDTH)
# r = mask.copy()
# g = mask.copy()
# b = mask.copy()
# for ll in range(0, n_classes):
#     r[mask == ll] = label_colours[ll, 0]
#     g[mask == ll] = label_colours[ll, 1]
#     b[mask == ll] = label_colours[ll, 2]
# rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
# rgb[:, :, 0] = r / 255.0
# rgb[:, :, 1] = g / 255.0
# rgb[:, :, 2] = b / 255.0
# plt.imshow(rgb)
# plt.show()
    
    
    
#img = imageio.imread("../input/pascal-voc-2012/VOC2012/SegmentationClass/2007_000323.png",pilmode="RGB")
# img = Image.open("../input/pascal-voc-2012/VOC2012/JPEGImages/2007_000032.jpg").convert('RGB')
# img = img.resize((400, 400))
# img = np.array(img)

# resize = transforms.CenterCrop((256, 256))
# img = resize(img)
# img = img.thumbnail(size)
# img = np.array(img)
# img = Image.fromarray(img)
# img.resize(size=(128, 128))
#img = imread("../input/pascal-voc-2012/VOC2012/SegmentationClass/2007_000032.png",as_gray=False)
#img = resize(img, (300, 300), mode='constant', preserve_range=True)

# plt.imshow(img)
# plt.show()


# label_colours = get_pascal_labels()
# single_img = Y_train[0]
# single_layer = np.argmax(single_img, axis=-1)
# r = single_layer.copy()
# g = single_layer.copy()
# b = single_layer.copy()
# for ll in range(0, n_classes):
#     r[single_layer == ll] = label_colours[ll, 0]
#     g[single_layer == ll] = label_colours[ll, 1]
#     b[single_layer == ll] = label_colours[ll, 2]
# rgb = np.zeros((single_layer.shape[0], single_layer.shape[1], 3))
# rgb[:, :, 0] = r /255.0
# rgb[:, :, 1] = g /255.0
# rgb[:, :, 2] = b /255.0
# plt.imshow(rgb)
# plt.show()
plt.imshow(X_train[0])
plt.show()
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
#building U-NET
inputs = Input((128, 128, 3))
s = Lambda(lambda x: x / 255.0) (inputs)

c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(21, (1, 1), activation='softmax') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# import tensorflow as tf

# from tensorflow_examples.models.pix2pix import pix2pix



# from IPython.display import clear_output
# import matplotlib.pyplot as plt

# OUTPUT_CHANNELS = 21
# base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
# # Use the activations of these layers
# layer_names = [
#     'block_1_expand_relu',   # 64x64
#     'block_3_expand_relu',   # 32x32
#     'block_6_expand_relu',   # 16x16
#     'block_13_expand_relu',  # 8x8
#     'block_16_project',      # 4x4
# ]
# layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
# down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

# down_stack.trainable = False
# up_stack = [
#     pix2pix.upsample(512, 3),  # 4x4 -> 8x8
#     pix2pix.upsample(256, 3),  # 8x8 -> 16x16
#     pix2pix.upsample(128, 3),  # 16x16 -> 32x32
#     pix2pix.upsample(64, 3),   # 32x32 -> 64x64
# ]
# def unet_model(output_channels):
#   inputs = tf.keras.layers.Input(shape=[128, 128, 3])
#   x = inputs

#   # Downsampling through the model
#   skips = down_stack(x)
#   x = skips[-1]
#   skips = reversed(skips[:-1])

#   # Upsampling and establishing the skip connections
#   for up, skip in zip(up_stack, skips):
#     x = up(x)
#     concat = tf.keras.layers.Concatenate()
#     x = concat([x, skip])

#   # This is the last layer of the model
#   last = tf.keras.layers.Conv2DTranspose(
#       output_channels, 3, strides=2,
#       padding='same')  #64x64 -> 128x128

#   x = last(x)

#   return tf.keras.Model(inputs=inputs, outputs=x)
# model = unet_model(OUTPUT_CHANNELS)
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# tf.keras.utils.plot_model(model, show_shapes=True)

model.fit(X_train, Y_train, batch_size=16, epochs=50)


model.save("validation_model2")

X_test = np.zeros((len(files["val"]), IMG_HEIGHT, IMG_WIDTH, 3), dtype=int)
Y_test= np.zeros((len(files["val"]), IMG_HEIGHT, IMG_WIDTH, 3), dtype=int)
pos = 0
for ii in tqdm(files["val"]):
    fname = ii + ".png"
    Jname = ii + ".jpg"
    jpg_path = pjoin('../input/pascal-voc-2012/VOC2012/', "JPEGImages", Jname)
    jpg_img = Image.open(jpg_path).convert('RGB')
    jpg_img = jpg_img.resize((IMG_HEIGHT, IMG_WIDTH))
    jpg_img = np.array(jpg_img)
    X_test[pos] = jpg_img
    
    lbl_path = pjoin('../input/pascal-voc-2012/VOC2012/', "SegmentationClass", fname)
    lbl_img = Image.open(lbl_path).convert('RGB')
    #resize for all the image
    lbl_img = lbl_img.resize((IMG_HEIGHT, IMG_WIDTH))
    lbl_img = np.array(lbl_img)
    Y_test[pos] = lbl_img
    pos = pos+1
plt.imshow(Y_test[10])
plt.show()
plt.imshow(X_test[10])
plt.show()
import tensorflow as tf
model = tf.keras.models.load_model("../output/kaggle/working/validation_model2")
Y_pre = model.predict(X_train)
print(Y_pre[0])
print (Y_pre.shape)
from numpy import argmax
inverted = np.argmax(Y_pre,axis=-1)
print(inverted)
label_colours = get_pascal_labels()
mask = inverted[10]
r = mask.copy()
g = mask.copy()
b = mask.copy()
for ll in range(0, n_classes):
    r[mask == ll] = label_colours[ll, 0]
    g[mask == ll] = label_colours[ll, 1]
    b[mask == ll] = label_colours[ll, 2]
rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
rgb[:, :, 0] = r 
rgb[:, :, 1] = g 
rgb[:, :, 2] = b 
plt.imshow(rgb)
plt.show()
plt.imshow(X_test[10])
plt.show()