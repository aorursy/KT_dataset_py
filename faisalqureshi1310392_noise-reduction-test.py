%matplotlib inline
import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk


# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

original = train[6][0]
#thresh = threshold_otsu(image)
#image = image > thresh
noisy_image = img_as_ubyte(original)
noise = np.random.random(noisy_image.shape)
noisy_image[noise > 0.99] = 255
noisy_image[noise < 0.01] = 0

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()

ax1.imshow(train[6][0], vmin=0, vmax=255, cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')
ax1.set_adjustable('box-forced')

ax2.imshow(median(noisy_image, disk(2)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax2.set_title('Median $r=1$')
ax2.axis('off')
ax2.set_adjustable('box-forced')
