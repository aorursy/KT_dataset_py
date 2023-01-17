# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from skimage import measure

import glob

import random

from scipy.misc import imread



import numpy as np

import matplotlib.pyplot as plt

from skimage.feature import blob_doh

from skimage.color import rgb2gray

from skimage.feature import CENSURE

from skimage.feature import (match_descriptors, corner_harris,

                             corner_peaks, ORB, plot_matches)
select = 1 # More later

# Load image

files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:select]

images = np.array([imread(img) for img in files])

image = images[0]

detector = CENSURE()
fig, ax = plt.subplots()

ax.axis('off')

ax.imshow(image)

plt.show()
image_gray = rgb2gray(image)

fig, ax = plt.subplots()

ax.axis('off')

ax.imshow(image_gray)

plt.show()
detector.detect(image_gray)

fig, ax = plt.subplots()

ax.axis('off')

ax.imshow(image_gray, cmap=plt.cm.gray)

ax.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],

              2 ** detector.scales, facecolors='none', edgecolors='r')

plt.show()
descriptor_extractor = ORB(n_keypoints=1000)

descriptor_extractor.detect_and_extract(image_gray)

keypoints1 = descriptor_extractor.keypoints

descriptors1 = descriptor_extractor.descriptors



fig, ax = plt.subplots()

plt.gray()

ax.axis('off')

ax.imshow(image_gray)

ax.scatter(descriptor_extractor.keypoints[:, 1], descriptor_extractor.keypoints[:, 0],

              2 ** detector.scales, facecolors='none', edgecolors='r')

plt.show()
image_blobs_doh = blob_doh(image_gray)

fig, ax = plt.subplots()

ax.axis('off')

ax.imshow(image_blobs_doh)

plt.show()