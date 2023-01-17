import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow as tf
from keras.optimizers import Adam
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
%matplotlib inline


# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
data_dir = "../input/train/"
mask_dir = "../input/train_masks/"
all_images_paths = os.listdir(data_dir)

# pick which images we will use for testing and which for validation
train_images_paths, validation_images_paths = train_test_split(all_images_paths, train_size=0.8, test_size=0.2)
def grey2rgb(img):
    new_image = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_image.append([img[i][j]]*3)
            
    new_image = np.array(new_image).reshape(img.shape[0], img.shape[1], 3)
    return new_image

def data_gen(data_dir, mask_dir,  img_paths, batch_size, dims):
    while True:
        index_list = np.random.choice(np.arange(len(img_paths)), batch_size)
        batch_imgs = []
        batch_labels = []
        for i in index_list:
            #img
            org_img = load_img(data_dir + img_paths[i])
            resized_img = imresize(org_img, dims+[3])
            array_img = img_to_array(resized_img)/255
            batch_imgs.append(array_img)
            
            #masks
            org_mask = load_img(mask_dir + img_paths[i].split('.')[0]+'_mask.gif')
            resized_img = imresize(org_img, dims+[3])
            array_mask = img_to_array(resized_img)/255
            batch_labels.append(array_mask[:,:,0])
            
        batch_imgs = np.array(batch_imgs)
        batch_labels = np.array(batch_labels).reshape(-1, dims[0], dims[1], 1)
        yield batch_imgs, batch_labels
        
train_gen = data_gen(data_dir, mask_dir, train_images_paths, 5, [128, 128])

img, msk = next(train_gen)

print(img.shape, msk.shape)
plt.imshow(img[0])
plt.show()
plt.imshow(grey2rgb(msk[0]), alpha=0.5)
plt.show()
def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu', )(input_layer)
    residual = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu', )(conv1)
    
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual
    
def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size = (2,2), padding = 'same')(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3,3), padding = 'same', activation = 'relu')(concat)
    conv2 = Conv2D(filters, (3,3), padding = 'same', activation = 'relu')(conv1)
    return conv2

filters = 64
input_layer = Input(shape = [128, 128, 3])
layers = [input_layer]
residuals = []

#down1 from 128px input and increase filters to 64
d1, res1 = down(input_layer, filters)
residuals.append(res1)

filters*=2 # double filter size for next unet operation

#down2 from 64px input and increase filters to 128
d2, res2 = down(d1, filters)
residuals.append(res2)

filters*=2 # double filter size for ...

#down3 ... filters to 256
d3, res3 = down(d2, filters)
residuals.append(res3)

filters*=2

#down4 ... filters to 512
d4, res4 = down(d3, filters)
residuals.append(res4)

filters*=2

#down5 ... filters to 1024
d5, res5 = down(d4, filters)
residuals.append(res5)

filters*=2
d6 = down(d5, filters, pool=False)

#up1 .. 512
up1 = up(d6, residuals[-1], filters//2)
filters//=2

#up2 .. 256
up2 = up(up1, residuals[-2], filters//2)
filters//=2

#up3 .. 128
up3 = up(up2, residuals[-3], filters//2)
filters//=2

#up4 .. 64
up4 = up(up3, residuals[-4], filters//2)
filters//=2

#up5 .. 128
up5 = up(up4, residuals[-5], filters//2)
filters//=2

out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up5)

model = Model(input_layer, out)
model.summary()
def dice_coeff(y_true, y_pred):
    smooth = 10**-5
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    isct = tf.reduce_sum(y_true*y_pred)
    
    return 2*isct/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred))
    
model.compile(optimizer = Adam(10**-4), loss='binary_crossentropy', metrics = [dice_coeff])
model.fit_generator(train_gen, steps_per_epoch=100, epochs=10)