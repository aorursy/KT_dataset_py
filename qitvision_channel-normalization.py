import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from scipy.misc import imread

import os

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Channel visualization
channel_names = ['Green','Red','Blue','Yellow']
channel_colors = ['mediumseagreen', 'salmon', 'steelblue', 'burlywood']
channel_cmaps = ['Greens','Reds','Blues','Oranges']

# Load training labels
train_labels = pd.read_csv("../input/train.csv")
print('Number of training images = {0}'.format(train_labels.shape[0]))
# set the path to our training image folder
train_path = "../input/train/"
# Helper function for loading images
# Copied from: https://www.kaggle.com/allunia/protein-atlas-exploration-and-baseline
def load_image(basepath, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = imread(basepath + image_id + "_green" + ".png")
    images[1,:,:] = imread(basepath + image_id + "_red" + ".png")
    images[2,:,:] = imread(basepath + image_id + "_blue" + ".png")
    images[3,:,:] = imread(basepath + image_id + "_yellow" + ".png")
    return images
for id in train_labels.Id:
    images = load_image(train_path, id)
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    for n in range(4):
        ax[n].imshow(images[n], cmap=channel_cmaps[n])
        ax[n].set_title(channel_names[n])
    break
def mean_from_histogram(arr):
    hist_sum = 0
    count = np.sum(arr)
    for n in range(len(arr)):
        hist_sum += n * arr[n]
    return hist_sum / count

def stdev_from_histogram(arr, mean):
    count = np.sum(arr)
    variance = 0
    for n in range(len(arr)):
        variance += arr[n] * (n - mean) * (n - mean)
    return np.sqrt(variance/count)
channel_hist = np.zeros(shape=(4,256))
channel_means = np.zeros(shape=(4))
channel_stdevs = np.zeros(shape=(4))

# These iterations are divided into two cells because the Jupyter Notebook cell timeout is 20 minutes
# and going through all images takes about 25 minutes
from tqdm import tqdm
counter = 0
for id in train_labels.Id:
    images = load_image(train_path, id)
    for n in range(4):
        hist, _ = np.histogram(images[n], 256, density=False)
        channel_hist[n,:,] = np.sum([hist,channel_hist[n]], axis=0)
    counter += 1
    if(counter == 15000):
        break
for id in train_labels.Id:
    if(counter > 0):
        counter -= 1
        continue
    images = load_image(train_path, id)
    for n in range(4):
        hist, _ = np.histogram(images[n], 256, density=False)
        channel_hist[n,:,] = np.sum([hist,channel_hist[n]], axis=0)
fig, ax = plt.subplots(1,4,figsize=(20,5))
x = range(256)
fig.suptitle('Histograms (log)', fontsize=16)

# Calculate means and standard deviations for each channel
for n in range(4):
    ax[n].bar(x, channel_hist[n], color=channel_colors[n], width=1.0)
    ax[n].set_yscale('log')
    ax[n].set_title(channel_names[n])
    channel_means[n] = mean_from_histogram(channel_hist[n])
    channel_stdevs[n] = stdev_from_histogram(channel_hist[n], channel_means[n])
    print(channel_names[n] + ': Mean = {0} ,StDev = {1}'.format(channel_means[n], channel_stdevs[n]))
    
plt.show()
import skimage.io
from skimage.transform import resize
def load_normalized_image(basepath, image_id):
        image_green = skimage.io.imread(basepath + image_id + "_green" + ".png")
        image_red = skimage.io.imread(basepath + image_id + "_red" + ".png")
        image_blue = skimage.io.imread(basepath + image_id + "_blue" + ".png")
        image_yellow = skimage.io.imread(basepath + image_id + "_yellow" + ".png")

        # normalize with calculated channel means and standard deviations
        image = np.stack((
            (image_red - 20.535) / 38.161,
            (image_green - 13.528) / 28.700,
            (image_blue - 14.249) / 40.195, 
            (image_yellow - 21.106) / 38.172), -1)
        
        image = resize(image, (512, 512, 4), mode='reflect')
        return image.astype(np.float)
def load_normalized_0_1_image(basepath, image_id):
        image_green = skimage.io.imread(basepath + image_id + "_green" + ".png")
        image_red = skimage.io.imread(basepath + image_id + "_red" + ".png")
        image_blue = skimage.io.imread(basepath + image_id + "_blue" + ".png")
        image_yellow = skimage.io.imread(basepath + image_id + "_yellow" + ".png")
        
        image = np.stack((
            image_red / 255.,
            image_green / 255.,
            image_blue / 255., 
            image_yellow / 255.), -1)
        
        image = resize(image, (512, 512, 4), mode='reflect')
        return image.astype(np.float)
counter = 0
# 256 bins
hist_100 = np.zeros(shape=(256))
hist_100_reg = np.zeros(shape=(256))
hist_100_0_1 = np.zeros(shape=(256))
bin_edges = []
bin_edges_reg = []
bin_edges_0_1 = []
for id in train_labels.Id:
    # load normalized (mean SD)
    image = load_normalized_image(train_path, id)
    hist, bin_edges = np.histogram(image, 256, (-1,10), density=False)
    hist_100 = np.sum([hist, hist_100], axis=0)
    # load normalized to 0-1 range
    image = load_normalized_0_1_image(train_path, id)
    hist, bin_edges_0_1 = np.histogram(image, 256, (0,1), density=False)
    hist_100_0_1 = np.sum([hist, hist_100_0_1], axis=0)
    # load regular
    images = load_image(train_path, id)
    for n in range(4):
        hist, bin_edges_reg = np.histogram(images[n], 256, density=False)
        hist_100_reg = np.sum([hist,hist_100_reg], axis=0)
    
    counter += 1
    if(counter == 100):
        break

fig, ax = plt.subplots(1,3,figsize=(18,5))
ax[0].bar(bin_edges_reg[1:], hist_100_reg, 1, color='steelblue')
#ax[0].set_yscale('log')
ax[0].set_ylim(0,1000000)
ax[0].set_title('Without normalization')

ax[1].bar(bin_edges[1:], hist_100, 0.043, color='mediumaquamarine')
#ax[1].set_yscale('log')
ax[1].set_ylim(0,1000000)
ax[1].set_title('Normalized ((channel - mean)/SD)')

ax[2].bar(bin_edges_0_1[1:], hist_100_0_1, 0.004, color='sandybrown')
#ax[2].set_yscale('log')
ax[2].set_ylim(0,1000000)
ax[2].set_title('Normalized (channel/255)')

fig.suptitle('Histogram (sum of all channels)', fontsize=16)
plt.show()