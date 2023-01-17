import numpy as np # linear algebra
import os
import matplotlib.pyplot as plot
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import random

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, LeakyReLU, UpSampling2D, concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint

import collections
from os.path import join as pjoin

import tensorflow as tf

from tensorflow import keras
def LoadImage(name, path_seg = "../input/pascal-voc-2012/VOC2012/SegmentationClass",
              path_img="../input/pascal-voc-2012/VOC2012/JPEGImages", cut_bottom=58,size=(128, 128)):
    img = Image.open(path_img+"/"+name+".jpg").convert("RGB")
    seg = Image.open(path_seg+"/"+name+".png").convert("RGB")
    img = np.array(img)
    seg = np.array(seg)
    
    
    img = Image.fromarray(img).resize(size)
    seg = Image.fromarray(seg).resize(size)
    
    
    img = np.array(img)
    seg = np.array(seg)


    return img/255, seg

#read the list of train, val, trainval in Segmentation folder
files=collections.defaultdict(list)
for split in ["train", "val"]:
    path = pjoin("/kaggle/input/pascal-voc-2012/VOC2012/ImageSets/Segmentation/", split + ".txt")
    file_list = tuple(open(path, "r"))
    file_list = [id_.rstrip() for id_ in file_list]
    files[split] = file_list

#test 
img, seg = LoadImage(files["val"][10])
plot.imshow(img)
plot.show()
plot.imshow(seg)
plot.show()
IMG_HEIGHT=128
IMG_WIDTH = 128
n_classes= 21
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
n_classes = 21
def LayersToRGBImage(img):
#     label_colours = get_pascal_labels()
#     r = mask.copy()
#     g = mask.copy()
#     b = mask.copy()
#     for ll in range(0, n_classes):
#         r[mask == ll] = label_colours[ll, 0]
#         g[mask == ll] = label_colours[ll, 1]
#         b[mask == ll] = label_colours[ll, 2]
#     rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
#     rgb[:, :, 0] = r 
#     rgb[:, :, 1] = g 
#     rgb[:, :, 2] = b 
#     return rgb/255.0
    colors = get_pascal_labels()
    nimg = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[2]):
        c = img[:,:,i]
        col = colors[i]
        
        for j in range(3):
            nimg[:,:,j]+=col[j]*c
    nimg = nimg/255.0
    return nimg

def ColorsToClass(seg):
    label_mask = np.zeros((seg.shape[0], seg.shape[1]), dtype=int)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(seg == label, axis=-1))[:2]] = ii
    cls = np.zeros((seg.shape[0], seg.shape[1],n_classes))
#     for i in range(IMG_HEIGHT):
#         for j in range(IMG_WIDTH):
#             lbl = label_mask[i][j]
#             encode_label_mask[i][j][lbl] = 1.0
    for i in range(n_classes):
        m = np.copy(label_mask)
        m[m!=i] = 0
        m[m!=0] = 1
        
        cls[:,:,i]=m
    return cls

#test
img, seg = LoadImage(files["val"][20])
seg2 = ColorsToClass(seg)

seg2 = LayersToRGBImage(seg2)

plot.imshow(seg2)
plot.show()
plot.imshow(img)
plot.show()
    

def Generate(path="train", batch_size=10):
    
    files_train = files[path] 
    while True:
        imgs=[]
        segs=[]
        
        for i in range(batch_size):
            file = random.sample(files_train,1)[0]
            
            
            img, seg = LoadImage(file)
            
            seg = ColorsToClass(seg)
            
            imgs.append(img)
            segs.append(seg)
        yield np.array(imgs), np.array(segs)
        
gen = Generate()
imgs, segs = next(gen)

plot.subplot(121)
plot.imshow(imgs[0])
plot.subplot(122)
plot.imshow(LayersToRGBImage(segs[0]))
plot.show()

# import os
# import sys
# import random
# import warnings

# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from itertools import chain


# from keras.models import Model, load_model
# from keras.layers import Input
# from keras.layers.core import Dropout, Lambda
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# #building U-NET
# inputs = Input((256, 256, 3))

# c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
# c1 = Dropout(0.1) (c1)
# c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
# p1 = MaxPooling2D((2, 2)) (c1)

# c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
# c2 = Dropout(0.1) (c2)
# c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
# p2 = MaxPooling2D((2, 2)) (c2)

# c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
# c3 = Dropout(0.1) (c3)
# c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
# p3 = MaxPooling2D((2, 2)) (c3)

# c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
# c4 = Dropout(0.1) (c4)
# c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
# p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

# c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
# c5 = Dropout(0.1) (c5)
# c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

# u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
# u6 = concatenate([u6, c4])
# c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
# c6 = Dropout(0.2) (c6)
# c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

# u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
# u7 = concatenate([u7, c3])
# c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
# c7 = Dropout(0.2) (c7)
# c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

# u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
# u8 = concatenate([u8, c2])
# c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
# c8 = Dropout(0.1) (c8)
# c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

# u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
# u9 = concatenate([u9, c1], axis=3)
# c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
# c9 = Dropout(0.1) (c9)
# c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

# outputs = Conv2D(21, (1, 1), activation='softmax') (c9)

# model = Model(inputs=[inputs], outputs=[outputs])



# opt = Adam(lr=0.0001)
# model.compile(optimizer=opt,
#              loss="categorical_crossentropy",
#              metrics=["accuracy"])
# model.summary()




inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

x1 = BatchNormalization()(inp)
x1 = Conv2D(64, 12, activation="relu", padding="same")(x1)
x1 = Conv2D(128, 12, activation="relu", padding="same")(x1)
p1 = MaxPooling2D()(x1)
#p1 = Dropout(0.2)(p1)

#x2 = BatchNormalization()(x1)
x2 = Conv2D(128, 9, activation="relu", padding="same")(p1)
x2 = Conv2D(128, 9, activation="relu", padding="same")(x2)
p2 = MaxPooling2D()(x2)
#p2 = Dropout(0.2)(p2)

#x3 = BatchNormalization()(x2)
x3 = Conv2D(128, 6, activation="relu", padding="same")(p2)
x3 = Conv2D(128, 6, activation="relu", padding="same")(x3)
p3 = MaxPooling2D()(x3)
#p3 = Dropout(0.2)(p3)

#x4 = BatchNormalization()(x3)
x4 = Conv2D(128, 3, activation="relu", padding="same")(p3)
x4 = Conv2D(128, 3, activation="relu", padding="same")(x4)
#x4 = MaxPooling2D()(x4)
#x4 = Dropout(0.2)(x4)

x5 = UpSampling2D()(x4)
x5 = concatenate([x3, x5])
x5 = Conv2D(128, 6, activation="relu", padding="same")(x5)
x5 = Conv2D(128, 6, activation="relu", padding="same")(x5)
#x5 = Dropout(0.2)(x5)

x6 = UpSampling2D()(x5)
x6 = concatenate([x2, x6])
x6 = Conv2D(128, 6, activation="relu", padding="same")(x6)
x6 = Conv2D(128, 6, activation="relu", padding="same")(x6)
#x6 = Dropout(0.2)(x6)

x7 = UpSampling2D()(x6)
x7 = concatenate([x1, x7])
x7 = Conv2D(21, 6, activation="relu", padding="same")(x7)
x7 = Conv2D(21, 6, activation="softmax", padding="same")(x7)



model = Model(inp, x7)
metric = tf.keras.metrics.MeanIoU(num_classes=21)

opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
             loss="categorical_crossentropy",
             metrics=[tf.keras.metrics.MeanIoU(num_classes=21)])
model.summary()

train_gen = Generate()
val_gen = Generate(path="val")
clb = [ModelCheckpoint("loss.h5", save_best_only=True, verbose=0)]

h = model.fit_generator(train_gen, epochs=500, steps_per_epoch=100,
                       
                       callbacks=clb, verbose=1)
model.save("model.h5")
model = load_model("loss.h5")