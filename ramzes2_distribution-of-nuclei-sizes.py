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
def getNucleiSize(image_id):

    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)

    masks = skimage.io.imread_collection(mask_file).concatenate()

    num_masks = masks.shape[0]

    nucleiSize = []

    for index in range(0, num_masks):

        nucleiSize.append(np.sum(masks[index] > 128))

    return np.mean(nucleiSize)



image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()

meanNucleiSizes = []

for image_id in image_ids:

    meanNucleiSizes.append(getNucleiSize(image_id))
print(np.min(meanNucleiSizes), np.median(meanNucleiSizes), np.max(meanNucleiSizes))

plt.hist(meanNucleiSizes, bins=20)
def find_nearest(array,value):

    idx = (np.abs(array-value)).argmin()

    return idx



def read_image_labels(image_id):

    from skimage import morphology

    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)

    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)

    image = skimage.io.imread(image_file)

    masks = skimage.io.imread_collection(mask_file).concatenate()    

    height, width, _ = image.shape

    num_masks = masks.shape[0]

    maxValue = np.max(image)

    kernel = np.matrix([[0,0,1,0,0],

                        [0,1,1,1,0],

                        [1,1,1,1,1],

                        [0,1,1,1,0],

                        [0,0,1,0,0] ], dtype=np.bool)

    for index in range(0, num_masks):

        contour = np.logical_xor(masks[index], morphology.binary_erosion(masks[index], kernel) )

#         image[:,:,0]=0

#         image[:,:,2]=0

        image[:,:,1] |= contour.astype(np.uint8)*255

        #image[contour > 0] = maxValue

    return image





def showImageWithNucleiSize(nucleiSize):

    print('nuclei size:', nucleiSize)

    i = find_nearest(np.asarray(meanNucleiSizes), nucleiSize)

    image_id = image_ids[i]

    plt.imshow(read_image_labels(image_id) )

    plt.show()

    

print('min size')

showImageWithNucleiSize(np.min(meanNucleiSizes))

print('mean size')

showImageWithNucleiSize(np.mean(meanNucleiSizes))

print('median size')

showImageWithNucleiSize(np.median(meanNucleiSizes))

print('max size')

showImageWithNucleiSize(np.max(meanNucleiSizes))

print('second peak on hist')

showImageWithNucleiSize(1500)
ids = np.argsort(meanNucleiSizes)[0:10]

for i in ids:

    image_id = image_ids[i]

    print(image_id, ' ', meanNucleiSizes[i])

    plt.imshow(read_image_labels(image_id))

    plt.show()

    