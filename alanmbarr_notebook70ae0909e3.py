import os

import numpy as np

import matplotlib.pyplot as plt

from skimage.transform import resize

import tensorflow as tf



def imcrop_tosquare(img):

    """Make any image a square image.



    Parameters

    ----------

    img : np.ndarray

        Input image to crop, assumed at least 2d.



    Returns

    -------

    crop : np.ndarray

        Cropped image.

    """

    if img.shape[0] > img.shape[1]:

        extra = (img.shape[0] - img.shape[1])

        if extra % 2 == 0:

            crop = img[extra // 2:-extra // 2, :]

        else:

            crop = img[max(0, extra // 2 - 1):min(-1, -extra // 2), :]

    elif img.shape[1] > img.shape[0]:

        extra = (img.shape[1] - img.shape[0])

        if extra % 2 == 0:

            crop = img[:, extra // 2:-extra // 2]

        else:

            crop = img[:, max(0, extra // 2 - 1):min(-1, -extra // 2)]

    else:

        crop = img

    return crop



dirnameTrain = "../input/train"

dirnameTest = "../input/test1"

def getFiles(dirName):

    return [os.path.join(dirName, fname) for fname in os.listdir(dirName)]



# Make sure we have exactly 100 image files!

filenames = getFiles(dirnameTrain)[:100]

imgs = [plt.imread(fname)[..., :3] for fname in filenames]

# Crop every image to a square

imgs = [imcrop_tosquare(img_i) for img_i in imgs]

# Then resize the square image to 100 x 100 pixels

imgs = [resize(img_i, (100, 100)) for img_i in imgs]# Finally make our list of 3-D images a 4-D array with the first dimension the number of images:

imgs = np.array(imgs).astype(np.float32)

print(imgs.shape)

sess = tf.InteractiveSession()

mean_img_op = tf.reduce_mean(imgs,2)

mean_img_4d = tf.reduce_mean(imgs, reduction_indices=0, keep_dims=True)

subtraction = imgs - tf.reduce_mean(imgs, reduction_indices=0, keep_dims=True)

std_img_op = tf.sqrt(tf.reduce_sum(subtraction * subtraction, reduction_indices=0))

mean_img = sess.run(mean_img_op)

std_img = sess.run(std_img_op)

norm_imgs_op = (imgs - mean_img) / std_img

norm_imgs = sess.run(norm_imgs_op)

print(np.min(norm_imgs), np.max(norm_imgs))

print(imgs.dtype)



sess.close()