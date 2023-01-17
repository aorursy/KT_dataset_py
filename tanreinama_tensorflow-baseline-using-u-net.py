# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sys

sys.path.append("../input/siim-acr-pneumothorax-segmentation")

from mask_functions import mask2rle, rle2mask
from glob import glob

dcm_files = {f.split("/")[-1]:f for f in list(glob('../input/siimacr-pneumothorax-segmentation-downloaded-file/siim/train/*/*/*.dcm'))\

             +list(glob('../input/siimacr-pneumothorax-segmentation-downloaded-file/siim/test/*/*/*.dcm'))}
import tensorflow as tf



class UNet:

    def __init__(self, size=(128, 128), l2_reg=None):

        self.model = self.create_model(size, l2_reg)



    @staticmethod

    def create_model(size, l2_reg, n_unit=64):

        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 1], name="inputs")

        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], 1], name="teacher")

        is_training = tf.placeholder(tf.bool, name="is_training")



        conv1_1 = UNet.conv(inputs, filters=n_unit, l2_reg_scale=l2_reg, istraining=is_training)

        conv1_2 = UNet.conv(conv1_1, filters=n_unit, l2_reg_scale=l2_reg, istraining=is_training)

        pool1 = UNet.pool(conv1_2)



        conv2_1 = UNet.conv(pool1, filters=n_unit//2, l2_reg_scale=l2_reg, istraining=is_training)

        conv2_2 = UNet.conv(conv2_1, filters=n_unit//2, l2_reg_scale=l2_reg, istraining=is_training)

        pool2 = UNet.pool(conv2_2)



        conv3_1 = UNet.conv(pool2, filters=n_unit//4, l2_reg_scale=l2_reg, istraining=is_training)

        conv3_2 = UNet.conv(conv3_1, filters=n_unit//4, l2_reg_scale=l2_reg, istraining=is_training)

        pool3 = UNet.pool(conv3_2)



        conv4_1 = UNet.conv(pool3, filters=n_unit//8, l2_reg_scale=l2_reg, istraining=is_training)

        conv4_2 = UNet.conv(conv4_1, filters=n_unit//8, l2_reg_scale=l2_reg, istraining=is_training)

        pool4 = UNet.pool(conv4_2)



        conv5_1 = UNet.conv(pool4, filters=n_unit//16, l2_reg_scale=l2_reg)

        conv5_2 = UNet.conv(conv5_1, filters=n_unit//16, l2_reg_scale=l2_reg)

        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=n_unit//16, l2_reg_scale=l2_reg), conv4_2], axis=3)



        conv_up1_1 = UNet.conv(concated1, filters=n_unit//8, l2_reg_scale=l2_reg)

        conv_up1_2 = UNet.conv(conv_up1_1, filters=n_unit//8, l2_reg_scale=l2_reg)

        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=n_unit//8, l2_reg_scale=l2_reg), conv3_2], axis=3)



        conv_up2_1 = UNet.conv(concated2, filters=n_unit//4, l2_reg_scale=l2_reg)

        conv_up2_2 = UNet.conv(conv_up2_1, filters=n_unit//4, l2_reg_scale=l2_reg)

        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=n_unit//4, l2_reg_scale=l2_reg), conv2_2], axis=3)



        conv_up3_1 = UNet.conv(concated3, filters=n_unit//2, l2_reg_scale=l2_reg)

        conv_up3_2 = UNet.conv(conv_up3_1, filters=n_unit//2, l2_reg_scale=l2_reg)

        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=n_unit//2, l2_reg_scale=l2_reg), conv1_2], axis=3)



        conv_up4_1 = UNet.conv(concated4, filters=n_unit, l2_reg_scale=l2_reg)

        conv_up4_2 = UNet.conv(conv_up4_1, filters=n_unit, l2_reg_scale=l2_reg)

        outputs = UNet.conv(conv_up4_2, filters=1, kernel_size=[1, 1], activation=None)



        return Model(inputs, outputs, teacher, is_training)



    @staticmethod

    def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, istraining=None):

        if l2_reg_scale is None:

            regularizer = None

        else:

            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)

        conved = tf.layers.conv2d(

            inputs=inputs,

            filters=filters,

            kernel_size=kernel_size,

            padding="same",

            activation=activation,

            kernel_regularizer=regularizer

        )

        if istraining is not None:

             conved = UNet.dropout(conved, istraining)



        return conved



    @staticmethod

    def dropout(inputs, is_training):

        droped = tf.layers.dropout(

            inputs=inputs,

            rate=0.1

        )

        return droped



    @staticmethod

    def pool(inputs):

        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)

        return pooled



    @staticmethod

    def conv_transpose(inputs, filters, l2_reg_scale=None):

        if l2_reg_scale is None:

            regularizer = None

        else:

            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)

        conved = tf.layers.conv2d_transpose(

            inputs=inputs,

            filters=filters,

            strides=[2, 2],

            kernel_size=[2, 2],

            padding='same',

            activation=tf.nn.relu,

            kernel_regularizer=regularizer

        )

        return conved



class Model:

    def __init__(self, inputs, outputs, teacher, is_training):

        self.inputs = inputs

        self.outputs = outputs

        self.teacher = teacher

        self.is_training = is_training

import cv2

import pydicom

def load_batch(indexs, df_load, train=True, imgsize=128):

    target_image = np.zeros((len(indexs),imgsize,imgsize,1),dtype=np.float32)

    target_mask = np.zeros((len(indexs),imgsize,imgsize,1),dtype=np.int32)

    original_size = []

    for ie,i in enumerate(indexs):

        file = dcm_files[df_load.iloc[i]["ImageId"]+".dcm"]

        img = pydicom.read_file(file).pixel_array

        stdim = np.std(img)

        menim = np.mean(img)

        img = np.clip(128 + 100 * (img - menim) / stdim, 0, 255)

        org_size = img.shape

        original_size.append(org_size)

        img = cv2.resize(img,(imgsize,imgsize),interpolation=cv2.INTER_NEAREST)

        target_image[ie,:,:,0] = img.astype(np.float32)

        if train:

            rle = df_load.iloc[i][" EncodedPixels"]

            if "-1" == str(rle).strip():

                mask = np.zeros((imgsize,imgsize))

            else:

                mask = rle2mask(rle,org_size[0],org_size[1])

                mask = cv2.resize(mask,(imgsize,imgsize),interpolation=cv2.INTER_NEAREST)

            # rotate mask image --- I seem this is correct for rle2mask. in siim sample, maybe are x and y swapped?

            mask = cv2.flip(cv2.warpAffine(mask, cv2.getRotationMatrix2D((imgsize//2,imgsize//2), 270, 1.0), (imgsize,imgsize), flags=cv2.INTER_LINEAR),1)

            target_mask[ie,:,:,0] = (mask != 0).astype(np.int32)

    return target_image, target_mask, original_size
from matplotlib import pyplot as plt

def show_train(imgsize = 128):

    df_train = pd.read_csv("../input/siimacr-pneumothorax-segmentation-downloaded-file/train-rle.csv")[:10]

    X, y, _ = load_batch(list(range(10)), df_train)

    for j in range(10):

            _x = X.astype(np.int32).reshape((-1,imgsize,imgsize))

            im = np.zeros((imgsize,imgsize*2))

            im[:,0:imgsize] = _x[j]

            im[:,imgsize:imgsize*2] = np.clip(_x[j]*0.5 + y[j].reshape((-1,imgsize,imgsize)) * 255, 0, 255)

            plt.imshow(im, cmap='bone')

            plt.axis('off')

            plt.show()

show_train(128)
from tqdm import tqdm_notebook as tqdm

model_unet = UNet(l2_reg=0.0001).model

loss_func = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=model_unet.teacher, logits=model_unet.outputs))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss_func)



def train(sess):

    df_train = pd.read_csv("../input/siimacr-pneumothorax-segmentation-downloaded-file/train-rle.csv")[:640]

    tf.global_variables_initializer().run()



    epochs = 20

    batch_size = 2



    for epoch in tqdm(range(epochs)):

        indexs = np.random.permutation(len(df_train))

        for i in range(len(df_train) // batch_size + 1):

            sp = i * batch_size

            ep = min(len(df_train), sp + batch_size)



            X, y, _ = load_batch(indexs[sp:ep], df_train)



            # Training

            sess.run(train_step, feed_dict={model_unet.inputs: X, model_unet.teacher: y, model_unet.is_training: True})
from matplotlib import pyplot as plt

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def pred(sess, spos=0.5):

    df_test = pd.read_csv("../input/siim-acr-pneumothorax-segmentation/sample_submission.csv")

    batch_size = 64

    imgsize = 128



    print("predict:")

    pred_masks = ["-1"] * len(df_test)

    for i in range(len(df_test) // batch_size + 1):

        sp = i * batch_size

        ep = min(len(df_test), sp + batch_size)



        X, _, org_size = load_batch(list(range(sp,ep,1)), df_test, False)



        y = sess.run(model_unet.outputs, feed_dict={model_unet.inputs: X, model_unet.is_training: False})

        y = sigmoid(y)

        y = ((y > spos).astype(np.int32) * 255).reshape((-1,imgsize,imgsize)).astype(np.uint8)

        if i == 0:

            for j in range(min(20,batch_size)):

                _x = X.reshape((-1,imgsize,imgsize)).astype(np.int32)

                im = np.zeros((imgsize,imgsize*2))

                im[:,0:imgsize] = _x[j]

                im[:,imgsize:imgsize*2] = np.clip(_x[j]*0.5 + y[j].reshape((-1,imgsize,imgsize)), 0, 255)

                plt.imshow(im, cmap='bone')

                plt.axis('off')

                plt.show()

        for j in range(len(y)):

            a = y[j].reshape((imgsize, imgsize, 1))

            s = org_size[j]

            mask = cv2.warpAffine(cv2.flip(a,1), cv2.getRotationMatrix2D((imgsize//2,imgsize//2), 90, 1.0), (imgsize,imgsize), flags=cv2.INTER_LINEAR)

            mask = cv2.resize(mask, s, interpolation=cv2.INTER_NEAREST)

            mask = mask2rle((mask!=0).astype(np.int32)*255, s[0], s[1])

            if len(mask) == 0:

                mask = "-1"

            pred_masks[sp + j] = mask

    df_test["EncodedPixels"] = pred_masks

    df_test.to_csv("submission.csv",index=False)
with tf.Session() as sess:

    train(sess)

    pred(sess, spos=0.2)