!cat /proc/cpuinfo
!free -h
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose

from keras.layers import BatchNormalization, Dropout, AveragePooling2D

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from keras.optimizers import Adam

# from tqdm import tqdm

import multiprocessing as mp

from multiprocessing import cpu_count

import os, cv2, time

from itertools import repeat

from functools import partial

from keras.preprocessing.image import load_img, ImageDataGenerator

%matplotlib inline
# set the necessary directories

data_dir = "../input/train/"

mask_dir = "../input/train_masks/"

test_dir = "../input/test/"

all_images = os.listdir(data_dir)

test_images = os.listdir(test_dir)

# %ls data
def preprocess(data_dir, img_name, dims, rles):

    img = load_img(data_dir+img_name)

    img = np.array(img, dtype='float32')/255.

    if rles:

        img = cv2.resize(img, (1918, 1280))

        mask = img > 0.5

        img = rle(mask)

    else:

        img = cv2.resize(img, dims)

    return img
# generator that we will use to read the data from the directory

def process_data(data_dir, mask_dir, batch_size, dims, images):

    """

    data_dir: where the actual images are kept

    mask_dir: where the actual masks are kept

    images: the filenames of the images we want to generate batches from

    batch_size: self explanatory

    dims: the dimensions in which we want to rescale our images

    """

    imgs = []

    labels = []

    # images

    img = preprocess(data_dir, images, dims, False)

    imgs.append(img)



    # masks

    mask = preprocess(mask_dir, images.split(".")[0] + '_mask.gif', dims, False)

    labels.append(mask[:, :, 0])

    return imgs, labels
def multicore_generator(images, batch_size=len(all_images)):

    ix = np.random.choice(np.arange(len(images)), batch_size) # from len(train_images) choose batch_size=64

    tic = time.time()

    pool = mp.Pool(processes=cpu_count())

    train_gen = partial(process_data, data_dir, mask_dir, batch_size, (256, 256))

    gen = pool.map_async(train_gen, list(np.array(images)[ix]), chunksize=8)

    gen.wait()

    results = gen.get()

    pool.close()

    pool.join()

    pool.terminate()

    x, y = zip(*results)

    x = np.array(x, dtype='float32').reshape(-1, 256, 256, 3)

    y = np.array(y, dtype='int32').reshape(-1, 256, 256, 1)

    print((time.time() - tic)/60.)

    return x, y
# Now let's use Tensorflow to write dice_coeficcient metric

def dice_coef(y_true, y_pred):

    smooth = 1e-5

    

    y_true = tf.round(tf.reshape(y_true, [-1]))

    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    

    isct = tf.reduce_sum(y_true * y_pred)

    

    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
# First let's define the two different types of layers that we will be using.

def down(input_layer, filters, pool=True):

    conv1 = Conv2D(filters, (3, 3), padding='same', activation='elu')(input_layer)

    conv2 = Conv2D(filters, (3, 3), padding='same', activation='elu')(conv1)

    residual = BatchNormalization(axis=3)(conv2)

    if pool:

        max_pool = MaxPool2D()(residual)

#         max_pool = AveragePooling2D()(residual)

        return max_pool, residual

    else:

        return residual



def up(input_layer, residual, filters):

    filters=int(filters)

    upsample = UpSampling2D()(input_layer)

    upconv = Conv2D(filters, (2, 2), padding="same")(upsample)

    concat = Concatenate(axis=3)([residual, upconv])

    drop = Dropout(0.25)(concat)

    conv1 = Conv2D(filters, (3, 3), padding='same', activation='elu')(drop)

    conv2 = Conv2D(filters, (3, 3), padding='same', activation='elu')(conv1)

    return conv2
# Make a custom U-nets implementation.

filters = 64

input_layer = Input(shape = [256, 256, 3])

layers = [input_layer]

residuals = []



# Down 1, 128

d1, res1 = down(input_layer, filters)

residuals.append(res1)



filters *= 2



# Down 2, 64

d2, res2 = down(d1, filters)

residuals.append(res2)



filters *= 2



# Down 3, 32

d3, res3 = down(d2, filters)

residuals.append(res3)



filters *= 2



# Down 4, 16

d4, res4 = down(d3, filters)

residuals.append(res4)



filters *= 2



# Down 5, 8

d5 = down(d4, filters, pool=False)



# Up 1, 16

up1 = up(d5, residual=residuals[-1], filters=filters/2)



filters /= 2



# Up 2,  32

up2 = up(up1, residual=residuals[-2], filters=filters/2)



filters /= 2



# Up 3, 64

up3 = up(up2, residual=residuals[-3], filters=filters/2)



filters /= 2



# Up 4, 128

up4 = up(up3, residual=residuals[-4], filters=filters/2)



out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)



model = Model(input_layer, out)

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])

model.summary()
def rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    bytes = np.where(img.flatten() == 1)[0]

    runs = []

    prev = -2

    for b in bytes:

        if (b > prev + 1): runs.extend((b + 1, 0))

        runs[-1] += 1

        prev = b



    return ' '.join([str(i) for i in runs])
def predict_masks(test_images, dims=(256, 256), batch_size=32):

    valid_imgs = []

    rles = []

    tic = time.time()

    pool = mp.Pool(processes=cpu_count())

    for batch in xrange(0, len(test_images), batch_size):

        resized = pool.map_async(preprocess, zip(repeat(test_dir), test_images[batch:batch+batch_size], 

                                                 repeat(dims), repeat(False)))

        resized.wait()

        predictions = model.predict_on_batch(np.array(resized.get()))

        valid_imgs.append(np.squeeze(predictions))

        masks = pool.map_async(preprocess, zip(repeat(test_dir), test_images[batch:batch+batch_size],

                                               repeat(dims), repeat(True)))

        masks.wait()

        rles.append(masks.get())

        

        

        print("{}:{}, {}, {}".format(batch,

                                     batch+batch_size, 

                                     len(test_images[batch:batch+batch_size]),

                                     np.array(resized.get()).shape, 

                                     len(masks.get())

                                )

         )

        

    pool.close()

    pool.join()

    pool.terminate()

    print("{} min.".format((time.time() - tic)/60.))
if __name__=="__main__":

    x, y = multicore_generator(all_images)

    print(x.shape, y.shape)

    model.fit(x, y, batch_size=12, epochs=10, validation_split=0.2)

    predictions = predict_masks(test_images)