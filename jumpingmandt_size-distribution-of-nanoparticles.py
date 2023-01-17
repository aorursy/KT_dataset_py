# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from PIL import Image

from skimage.exposure import histogram



from skimage import data

from skimage.filters import threshold_otsu

from skimage.segmentation import clear_border

from skimage.measure import label, regionprops

from skimage.morphology import closing, square

from skimage.color import label2rgb



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
im = Image.open('/kaggle/input/nanoparticles/sample_1.tif') 

imarray = np.array(im) 

imarray.shape 

plt.imshow(imarray,cmap = 'gray')
# filter the smaller dust

coins = np.array(im)

hist, hist_centers = histogram(coins)



fig, axes = plt.subplots(1, 2, figsize=(8, 3))

axes[0].imshow(coins, cmap=plt.cm.gray)

axes[0].axis('off')

axes[1].plot(hist_centers, hist, lw=2)

axes[1].set_title('histogram of gray values')
# set a threshold

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)



axes[0].imshow(imarray > 100, cmap=plt.cm.gray)

axes[0].set_title('coins > 100')



axes[1].imshow(imarray > 130, cmap=plt.cm.gray)

axes[1].set_title('coins > 130')



for a in axes:

    a.axis('off')



plt.tight_layout()
image = imarray > 100



# apply threshold

thresh = threshold_otsu(image)

bw = closing(image > thresh, square(4))



# remove artifacts connected to image border, or you can keep the particles at the boarder

# cleared = clear_border(bw)

cleared = bw



# label image regions

label_image = label(cleared)

# to make the background transparent, pass the value of `bg_label`,

# and leave `bg_color` as `None` and `kind` as `overlay`

image_label_overlay = label2rgb(label_image, image=image, bg_label=0)



fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(image_label_overlay)



arealist = []

for region in regionprops(label_image):

    # take regions with large enough areas

    if region.area >= 200:

        # draw rectangle around segmented coins

        minr, minc, maxr, maxc = region.bbox

        arealist.append(region.area)

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                                  fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)



ax.set_axis_off()

plt.tight_layout()

plt.show()
# open the txt file and get the information of pixel to nanometer



path_txt = '/kaggle/input/nanoparticles/detail_sample_1.txt'

txt_info = open(path_txt, "r")

info = txt_info.read().split()
# info of the sample name

info[6]
# info of the image size

info[11]
# info of the pixel size 1 pixel corresponds to how many nanometer

info[12]
# info of the pixel size 1 pixel corresponds to how many nanometer

pixelsize = float(info[12][10:])

pixelsize
# convert the unit from pixel to nanometer, 1 pixel = 0.9921875 nm



unit = pixelsize

sizelist =  [2*unit*(np.sqrt(element))/np.pi for element in arealist]
# plot the histogram of particle size distribution

fig, axes = plt.subplots(1, 2, figsize=(8, 3))



axes[0].imshow(imarray > 100, cmap=plt.cm.gray)

axes[0].set_title('SEM image of SnO2 Nanoparticles')

axes[0].axis('off')



axes[1].hist(sizelist,bins = 20, lw=2)

axes[1].set_title('histogram of NP diameter distribution')





plt.tight_layout()