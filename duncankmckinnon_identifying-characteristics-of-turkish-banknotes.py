import glob

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import random as rand



DATASET_ROOT_PATH = "/kaggle/input/turkish-lira-banknote-dataset"



# Get all images

files = sorted(glob.glob(os.path.join(DATASET_ROOT_PATH, "**/*.png")))
# number of samples of each image to show

n = 10



# sample of indices for images to show

rand.seed(2020)

rsample = rand.sample(range(1000), n)



# setup levels and breaks

levels = [10, 100, 20, 200, 5, 50]

image_idxs = [0, 1000, 2000, 3000, 4000, 5000]



# plot n samples of each level 

fig, axes = plt.subplots(6, n, figsize = (15,n))

plt.setp(axes.flat, xticks=[], yticks=[])



for i in range(n):

    j = 0

    for image_idx in image_idxs:

        file = files[(image_idx + rsample[i])]

        img = mpimg.imread(file)

        axes[j][i].imshow(img)

        j += 1

        

# add labels to levels

for ax,level in zip(axes[:,0], levels):

    ax.set_ylabel("₺{0}".format(level))

# get parameters for image size to use in averaging pixel values

imshape = mpimg.imread(files[0]).shape

imsize = imshape[0] * imshape[1]



# colors used in graphs

color_rgb = ['red','green','blue']



def avgRGBsForImg(img_num):

    """ Get the average color channels for an image

    Parameters

    ----------

    img_num: the index of the image in the dataset

    

    Returns

    ----------

    a list of the average pixel values for [ Red, Green, Blue ] channels

    """

    img = mpimg.imread(files[img_num])

    return [np.sum(img[:,:,i]) / imsize for i in range(3)]



def colorDistByClass(classnum, size):

    """Get all color distributions for a label class

    Parameters

    ----------

    classnum: the label class (values from 0:5)

    size: number of samples to pull from the class (0:1000)

    

    Returns

    ----------

    a list of the red, green and blue channel averages for each image in the class [R,G,B]

    """

    n = 1000 * ( classnum )

    [R,G,B] = [[],[],[]]

    for i in range(size):

        [r,g,b] = avgRGBsForImg(n + i)

        R.append(r)

        G.append(g)

        B.append(b)

    return [R,G,B]



def plotColorDistByClass(RGB, ax, label, bins = 30):

    """ Plot the color distributions for a given class

    Parameters

    ----------

    RGB: the color averages for the given class

    ax: the plot axis object

    label: the class label

    bins: the number of bins in the histogram

    """

    ax.hist(RGB, bins, density=True, histtype='bar', color = color_rgb, label = color_rgb)

    ax.set_title("₺{0} Banknote".format(label))

    

# store class distributions

RGBs = {}



# Plot color distributions by class

fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize=(15,10), sharex = True, sharey = True)



m = 0

for i in range(3):

    for j in range(2):

        RGB = colorDistByClass(m, 100)

        label = levels[m]

        plotColorDistByClass(RGB, ax[i][j], label)

        RGBs[m] = RGB

        m += 1

plt.show()
for i in range(6):

    channels = RGBs[i]

    print("\n₺{0} Banknote".format(levels[i]))

    for j in range(3):

        print('\t{} channel: \n\t\tmean:{}, \n\t\tstandard deviation:{}'.format(color_rgb[j], np.mean(channels[j]), np.std(channels[j])))