%matplotlib inline
import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier


# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

image = train[5][0]
thresh = threshold_otsu(image)
image = image > thresh
noisy_image = img_as_ubyte(image)
noise = np.random.random(noisy_image.shape)
noisy_image[noise > 0.99] = 255
noisy_image[noise < 0.01] = 0
image = median(noisy_image, disk(5))

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()

ax1.imshow(noisy_image, vmin=0, vmax=255, cmap=plt.cm.gray)
ax1.set_title('Noisy image')
ax1.axis('off')
ax1.set_adjustable('box-forced')

ax2.imshow(median(noisy_image, disk(1)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax2.set_title('Median $r=1$')
ax2.axis('off')
ax2.set_adjustable('box-forced')
