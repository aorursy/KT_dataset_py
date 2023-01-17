# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import skimage.io

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from skimage import morphology



def read_image_labels(image_id):

    # most of the content in this function is taken from 'Example Metric Implementation' kernel 

    # by 'William Cukierski'

    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)

    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)

    image = skimage.io.imread(image_file)

    masks = skimage.io.imread_collection(mask_file).concatenate()    

    height, width, _ = image.shape

    num_masks = masks.shape[0]

    maxValue = np.max(image)

    for index in range(0, num_masks):

        contour = np.logical_xor(masks[index], morphology.binary_erosion(masks[index]) )

        image[contour > 0] = maxValue

    return image



def plot_images_masks(image_ids):

    plt.close('all')

    fig, ax = plt.subplots(nrows=len(image_ids),ncols=1, figsize=(256,256))

    for ax_index, image_id in enumerate(image_ids):

        image = read_image_labels(image_id)

        ax[ax_index].imshow(image)



image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()

print("Total Images in Training set: {}".format(len(image_ids)))

random_image_ids = random.sample(image_ids, 16)

print("Randomly Selected Images: {}, their IDs: {}".format(len(random_image_ids), random_image_ids))

plot_images_masks(random_image_ids)

    