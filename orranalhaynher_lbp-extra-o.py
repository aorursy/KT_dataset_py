import numpy as np 

import pandas as pd

from glob import glob

from skimage import feature

from skimage.io import imread

import matplotlib.pyplot as plt

from sklearn import preprocessing

from skimage.exposure import histogram

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
def get_order_filenames(file_path):

    df = pd.read_csv(file_path)

    return df["image"]
file_path = '../input/specialist-segmentation/reading_order.csv'

image_names = get_order_filenames(file_path)
path = '../input/kmeanssegmentation/'

images = [imread(path+str(name)+'.bmp') for name in image_names]

print('The database has {} segmented images'.format(len(images)))
images = [im for im in images]
images[0].shape
np.max(images[0])
lbp = []

hist = []

for i, img in enumerate(images):

    lbp.append(feature.local_binary_pattern(img, 10, 1, method="default"))

    histFeat, bin_edges = np.histogram(lbp[i], range(256), density=True)

    hist.append(histFeat)
plt.imshow(lbp[0], cmap='gray')
hist[0]
np.max(hist[0])
bin_edges
np.save('./lbpKmeans', hist)
np.max(hist)