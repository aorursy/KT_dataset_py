import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

from tqdm import tqdm_notebook
img_size_ori = 101
img_size_target = 101

train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")

train_df = train_df.join(depths_df)

test_df = depths_df[~depths_df.index.isin(train_df.index)]
len(test_df)
train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=False)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
# Simple split of images into training and testing sets
ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 3), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.1, random_state=1234 )
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.applications.mobilenet import MobileNet
from keras.applications import Xception, InceptionResNetV2
from keras import optimizers
base_model = Xception( include_top=False, input_shape=((101,101,3)))
def conv_block_simple(prevlayer, filters, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides)(prevlayer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv
def build_model(start_neurons=16):
    
    for l in base_model.layers:
        l.trainable = True
    #--------------------------------------------------------------------------------------
    conv0 = base_model.get_layer('block1_conv1_act').output # 50
    conv1 = base_model.get_layer('block2_sepconv2_bn').output # 48
    conv2 = base_model.get_layer('block3_sepconv2_bn').output # 24
    conv3 = base_model.get_layer('block4_sepconv2_bn').output # 12
    conv4_1 = base_model.get_layer('block5_sepconv1').output # 6
    conv4 = base_model.get_layer('block13_sepconv2_bn').output # 6
    conv5 = base_model.get_layer('conv2d_4').output # 3 ----- ебанашка керас, не норм. имя, всегда разное
    conv6 = base_model.get_layer('block14_sepconv2_act').output # 3
    
    midlle = concatenate([conv5, conv6], axis=-1)
    convm = conv_block_simple(midlle, start_neurons * 16)
    convm = conv_block_simple(convm, start_neurons * 16)
    
    deconv1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv1 = concatenate([deconv1, conv4])
    deconv1 = conv_block_simple(deconv1, start_neurons * 8)
    deconv1 = conv_block_simple(deconv1, start_neurons * 8)
    deconv1 = concatenate([deconv1, conv4_1])
    deconv1 = conv_block_simple(deconv1, start_neurons * 8)
    
    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv1)
    deconv2 = concatenate([deconv2, conv3])
    deconv2 = conv_block_simple(deconv2, start_neurons * 4)
    deconv2 = conv_block_simple(deconv2, start_neurons * 4)
    
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
    deconv3 = concatenate([deconv3, conv2])
    deconv3 = conv_block_simple(deconv3, start_neurons * 4)
    deconv3 = conv_block_simple(deconv3, start_neurons * 4)
    
    deconv4 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv4 = concatenate([deconv4, conv1])
    deconv4 = conv_block_simple(deconv4, start_neurons * 2)
    deconv4 = conv_block_simple(deconv4, start_neurons * 2)
    
    deconv5 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(1, 1), padding="valid")(deconv4)
    deconv5 = concatenate([deconv5, conv0])
    deconv5 = conv_block_simple(deconv5, start_neurons * 1)
    deconv5 = conv_block_simple(deconv5, start_neurons * 1)    
    
    inp = base_model.input
    deconv6 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(deconv5)
    deconv6 = concatenate([deconv6, inp])
    deconv6 = conv_block_simple(deconv6, start_neurons * 1)
    deconv6 = conv_block_simple(deconv6, start_neurons * 1)     
    
    output = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv6)
    
    return Model(base_model.input, output)

model = build_model(start_neurons=4)



















