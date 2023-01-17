import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt
from pathlib import Path
import os
print(os.listdir("../input"))

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
train = pd.read_csv('../input/train_ship_segmentations.csv')
train.head()
sample = train[~train.EncodedPixels.isna()].sample(25)

fig, ax = plt.subplots(5, 5, sharex='col', sharey='row')
fig.set_size_inches(20, 20)

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    path = Path('../input/train') / '{}'.format(imgid)
    img = imread(path)
    
    ax[row, col].imshow(img)
#fig, ax = plt.subplots(5, 5, sharex='col', sharey='row')
#fig.set_size_inches(20, 20)
from PIL import Image, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    path = Path('../input/train') / '{}'.format(imgid)
    img = imread(path)
    img =img[:,:,1]
    radius = 15
    selem = disk(radius)

    local_otsu = rank.otsu(img, selem)
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    plt.tight_layout()

    fig.colorbar(ax[0].imshow(img, cmap=plt.cm.gray),
                 ax=ax[0], orientation='horizontal')
    ax[0].set_title('Original')
    ax[0].axis('off')

    fig.colorbar(ax[1].imshow(local_otsu, cmap=plt.cm.gray),
                 ax=ax[1], orientation='horizontal')
    ax[1].set_title('Local Otsu (radius=%d)' % radius)
    ax[1].axis('off')

    ax[2].imshow(img >= local_otsu, cmap=plt.cm.gray)
    ax[2].set_title('Original >= Local Otsu' % threshold_global_otsu)
    ax[2].axis('off')

    ax[3].imshow(global_otsu, cmap=plt.cm.gray)
    ax[3].set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
    ax[3].axis('off')

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    path = Path('../input/train') / '{}'.format(imgid)
    img = imread(path)
    im =img[:,:,1]
    #im = ndi.gaussian_filter(img, 4)


    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im, sigma=0.5)
    edges2 = feature.canny(im, sigma=2)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    path = Path('../input/train') / '{}'.format(imgid)
    print(path)
    img = imread(path)
    

    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
    print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
    print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(img, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(mark_boundaries(img, segments_slic))
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    ax[1, 1].set_title('Compact watershed')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template
img = imread('../input/train/15b4a80a3.jpg')
boat1 = img[ 40:140, 370:520,:]
img = imread('../input/train/fad674252.jpg')
boat3 = img[ 270:310, 640:740,:]


img = imread('../input/train/c62733fc9.jpg')
boat2 = img[ 220:250, 375:475,:]


fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(boat1)
ax2.imshow(boat2)
ax3.imshow(boat3)

plt.show()
    
for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    path = Path('../input/train') / '{}'.format(imgid)
    print(path)
    img = imread(path)
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(boat1)
    ax1.set_axis_off()
    ax1.set_title('template')
    
    result = match_template(img, boat1)
    result2 = match_template(img, boat2)
    result3 = match_template(img, boat3)
    
    ij = np.unravel_index(np.argmax(result), result.shape)
    print(ij)
    x, y,z = ij
    
    ij2 = np.unravel_index(np.argmax(result2), result2.shape)
    print(ij2)
    x2, y2,z = ij2
    
    ij3 = np.unravel_index(np.argmax(result3), result3.shape)
    print(ij3)
    x3, y3,z = ij3
    
    ax2.imshow(img, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin, z = boat1.shape
    rect = plt.Rectangle((y,x), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    
    rect2 = plt.Rectangle((y2,x2), wcoin, hcoin, edgecolor='b', facecolor='none')
    ax2.add_patch(rect2)

    rect3 = plt.Rectangle((y3,x3), wcoin, hcoin, edgecolor='g', facecolor='none')
    ax2.add_patch(rect3)
    result=img[ x:x+wcoin, y:y+hcoin,:]
    ax3.imshow(result)
    ax3.set_axis_off()
    #ax3.set_title('`match_template`\nresult')
    # highlight matched region
    #ax3.autoscale(False)
    #ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()
sample = train[train.EncodedPixels.isna()].sample(25)

fig, ax = plt.subplots(5, 5, sharex='col', sharey='row')
fig.set_size_inches(20, 20)

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    path = Path('../input/train') / '{}'.format(imgid)
    img = imread(path)
    
    ax[row, col].imshow(img)
train.groupby(train.EncodedPixels.isna()).size().plot(kind='bar', figsize=(12, 8));
def get_img(imgid):
    '''Return image array, given ID.'''
    path = Path('../input/train/') / '{}'.format(imgid)
    return imread(path)
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
fig.set_size_inches(20, 6)

mask = train.EncodedPixels.isna()
for i, (msk, label) in enumerate(zip([mask, ~mask], ['No Ships', 'Ships'])):
    _ids = train[msk].ImageId.sample(250)
    imgs = np.array([get_img(_id) for _id in _ids])
    
    red = imgs[:, :, :, 0]
    green = imgs[:, :, :, 1]
    blue = imgs[:, :, :, 2]
    
    ax[i].plot(np.bincount(red.ravel()), color='orangered', label='red', lw=2)
    ax[i].plot(np.bincount(green.ravel()), color='yellowgreen', label='green', lw=2)
    ax[i].plot(np.bincount(blue.ravel()), color='skyblue', label='blue', lw=2)
    ax[i].legend()
    ax[i].title.set_text(label)
