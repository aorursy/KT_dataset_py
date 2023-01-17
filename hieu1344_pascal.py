# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import json

import matplotlib.pyplot as plt

import cv2

from PIL import Image, ImageDraw, ImageFont

from matplotlib import patches, patheffects

from collections import defaultdict

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from pathlib import Path

path = Path('/kaggle/input/pascal-voc-2007/')

img_path = path / 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'

anno_path = path / 'PASCAL_VOC/PASCAL_VOC'
list(anno_path.iterdir())
train = json.load((anno_path/'pascal_train2007.json').open())

train.keys()
train['images'][:5]
train['annotations'][:2]
train['categories'][:10]
train_fn = {k['id']:k['file_name'] for k in train['images']}

train_wh = {k['id']:(k['width'], k['height']) for k in train['images']}

train_id = [k['id'] for k in train['images']]

cats = {c['id']:c['name'] for c in train['categories']}
def hw_bb(bbox):

    return np.array([bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]])



def bb_hw(bbox):

    return np.array([bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]])
train_anno = defaultdict(lambda: [])

for o in train['annotations']:

    bbox = o['bbox']

#     bbox = hw_bb(bbox)

    train_anno[o['image_id']].append((bbox, o['category_id']))

len(train_anno)
#Bounding box [x, y, width, height] (x, y) is the top left corner

def load_img(path, normalize=False):

    img = Image.open(path)

    img = np.asarray(img)

    return img/255 if normalize else img



def show_img(img, ax=None, figsize=(4,4)):

    if not ax:

        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img)

    ax.axis('off')

    return ax



def draw_outline(o, lw):

    o.set_path_effects([patheffects.Stroke(

        linewidth=lw, foreground='black'), patheffects.Normal()])



def draw_rect(ax, b, text=None):

    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))

    draw_outline(patch, 4)

    if text:

        text = ax.text(b[0]+1, b[1], text, verticalalignment='top',fontsize=14, weight='bold', color='white')

        draw_outline(text, 2)

        

def draw_img(img, anno):

    ax = show_img(img)

    for bb, c in anno:

        draw_rect(ax, bb, text=cats[c])

        

def draw_idx(idx):

    img = load_img(img_path/train_fn[idx])

    anno = train_anno[idx]

    draw_img(img, anno)
train_anno[train_id[55]]
img = load_img(img_path/train_fn[train_id[0]])

# img = img / 255

# anno = train_anno[train_id[1]]

# w, h = img.shape[1], img.shape[0]

# x_scale = 224 / w

# y_scale = 224 / h

# # img = tf.image.resize_with_pad(img, 224, 224, )

# # ax = show_img(img)

# # for bbox, c in anno:

# #     print(bbox, cats[c])

# #     b = bbox.copy()

# #     b[0], b[2] = b[0] * x_scale, b[2] * x_scale

# #     b[1], b[3] = b[1] * y_scale, b[3] * y_scale

# #     draw_rect(ax, b, text=cats[c])

# bbox = anno[1][0].copy()

# bbox[0], bbox[2] = bbox[0] * x_scale, bbox[2] * x_scale

# bbox[1], bbox[3] = bbox[1] * y_scale, bbox[3] * y_scale

# # img =cv2.resize(img, (224, 224))

# img = tf.image.resize_with_pad(img, 224, 224, )

# # print(img.shape)

# ax = show_img(img)

# draw_rect(ax, bbox)

# # draw_idx(train_id[55])

def get_largest(bbs):

    bbs = sorted(bbs, reverse=True,  key=lambda bb: np.prod(bb[0][2:]))

    return bbs[0]
train_lg_anno = {k:get_largest(v) for k, v in train_anno.items()}
idx = train_id[0]

bb, c = train_lg_anno[idx]

ax = show_img(load_img(img_path/train_fn[idx]))

draw_rect(ax, bb, cats[c])
class ResBlock(keras.layers.Layer):

    def __init__(self, filters, expansion, strides=1, skip=False,  activation='relu', **kwargs):

        super().__init__(**kwargs)

        self.activation = tf.keras.activations.get(activation)

        self.main_layer = []

        if expansion > 1:



            self.main_layer = [

                                Conv2D(filters, 1, strides=1, padding='same', kernel_initializer='he_normal',use_bias=False),

                                BatchNormalization(), 

                                Activation('relu'), 

                                Conv2D(filters, 3,strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False),

                                BatchNormalization(),

                                Activation('relu'), 

                                Conv2D(filters*expansion, 1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False),

                                BatchNormalization(gamma_initializer='zero')



            ]

        else:

            self.main_layer = [

                                Conv2D(filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal'),

                                BatchNormalization(),

                                Activation('relu'),

                                Conv2D(filters, 3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False),

                                BatchNormalization(gamma_initializer='zero')

        ]

        self.skip = []

        if strides > 1 or skip:

            self.skip = [

                            Conv2D(filters*expansion, 1, strides=strides, padding='same', use_bias=False),

                            BatchNormalization()

            ]

            

            

    def call(self, inputs):

        X = inputs

        for layer in self.main_layer:

            X = layer(X)

        skip_path = inputs

        for layer in self.skip:

            skip_path = layer(skip_path)

        return self.activation(X + skip_path)

            

    
def xresnet34(input_shape):

    input_ = Input(shape=input_shape)

#     X = input_

    nfs = [32, 64, 64]

    for i, f in enumerate(nfs):

        X = Conv2D(f, 3, padding='same', kernel_initializer='he_normal', use_bias=False, strides=2 if i==0 else 1)(input_ if i==0 else X)

        X = BatchNormalization()(X)

        X = Activation('relu')(X)





    X = MaxPooling2D((3, 3), strides=2, padding='same')(X)

    prev_filters = 64

    for i, filters in enumerate([64]*3 + [128]*4 + [256]*6 + [512]*3):

        strides = 1

        if filters != prev_filters:

            strides = 2

        X = ResBlock(filters, 1, strides=strides)(X)

        prev_filters = filters

    X = Flatten()(X)

    X = Dropout(0.5)(X)

    X = Dense(256, kernel_initializer='he_normal')(X)

    X = ReLU()(X)

    X = BatchNormalization()(X)

    X = Dropout(0.5)(X)

    bbox_out = Dense(4, kernel_initializer='he_normal', name='bbox_output')(X)

    cls_out = Dense(20, kernel_initializer='he_normal',  activation='softmax',name='class_output')(X)

    model = keras.models.Model(inputs=input_, outputs=[bbox_out, cls_out])

    return model
images = [str(img_path/train_fn[id]) for id in train_id]
bb_label =[train_lg_anno[id][0] for id in train_id]

cls_label = [train_lg_anno[id][1]-1 for id in train_id]
for i in range(len(bb_label)):

    bb_label[i] = np.asarray(bb_label[i])

    x_scale = 224 / train_wh[train_id[i]][0]

    y_scale = 224 / train_wh[train_id[i]][1]

    bb_label[i][0], bb_label[i][2] =   bb_label[i][0] * x_scale, bb_label[i][2] * x_scale

    bb_label[i][1], bb_label[i][3] =   bb_label[i][1] * y_scale, bb_label[i][3] * y_scale
bb_label
dataset = tf.data.Dataset.from_tensor_slices((images, bb_label ,cls_label))

dataset = dataset.map(lambda x, bb, cls: (x, (bb, cls)))
x, lbl = list(iter(dataset))[0]

x = tf.image.decode_jpeg(tf.io.read_file(x), channels=3)

x = tf.cast(x, dtype = tf.float32) / 255.

x = tf.image.resize(x, (224, 224), antialias=True, method='nearest')

ax = show_img(x)

draw_rect(ax, lbl[0].numpy(), cats[lbl[1].numpy()+1])

def preprocess(x, lbl):

    x = tf.image.decode_jpeg(tf.io.read_file(x), channels=3)

    x = tf.cast(x, dtype = tf.float32)

    x = tf.image.resize(x, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x = x / 255.

    return x, lbl
dataset = dataset.map(preprocess, num_parallel_calls=-1)
list(next(dataset.as_numpy_iterator()))[1]
dataset = dataset.shuffle(512).batch(64, drop_remainder=True).repeat().prefetch(32)
model = xresnet34((224, 224, 3))

model.summary()
model.compile(optimizer=keras.optimizers.Adam(1e-2),

              loss={'bbox_output': keras.losses.MeanAbsoluteError(),

                    'class_output':keras.losses.SparseCategoricalCrossentropy()},

              metrics={'class_output':tf.keras.metrics.SparseCategoricalAccuracy()},

             loss_weights={'bbox_output':1, 'class_output':1000})
model.fit(dataset, epochs=20, steps_per_epoch=39)
x, y = next(iter(dataset))
idx = np.random.randint(0, 64, 16)

fig, axs = plt.subplots(4, 4, figsize=(13, 13))

imgs = x.numpy()[idx]

bbs = y[0].numpy()[idx]

cls = y[1].numpy()[idx]

for i, ax in enumerate(axs.flat):

    show_img(imgs[i], ax)

    draw_rect(ax, bbs[i], cats[cls[i] + 1])

plt.tight_layout()



# img = load_img(img_path/train_fn[train_id[idx]]) / 255

# img = tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# img = load_img(img_path/train_fn[train_id[idx]]) / 255

# img = tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# bb = bb_label[idx]

# cls = cls_label[idx]

# ax = show_img(img)

# draw_rect(ax, bb, cats[cls+1]); bb
# pred = model.predict(tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[None, ...])

pred = model.predict(imgs)
pred
# c = pred[1].argmax() + 1

c = pred[1].argmax(axis=1) + 1

c
pred[0].shape
fig, axs = plt.subplots(4, 4, figsize=(13, 13))

for i, ax in enumerate(axs.flat):

    show_img(imgs[i], ax)

    draw_rect(ax, pred[0][i], cats[c[i]])

plt.tight_layout()